package server

import (
	"context"
	"io"
	"time"

	"github.com/google/uuid"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	enginepb "llm-server/gen/engine"
	gatewaypb "llm-server/gen/gateway"
)

// engineClient is the subset of engine.Client that the gRPC handler needs.
// Defined here (consumer side) so tests can substitute an in-process fake
// gRPC server instead of dialing a real C++ engine process.
type engineClient interface {
	Submit(ctx context.Context, req *enginepb.EngineRequest) (enginepb.InferenceEngine_SubmitRequestClient, error)
	GetMetrics(ctx context.Context) (*enginepb.EngineMetrics, error)
}

type Server struct {
	gatewaypb.UnimplementedInferenceServiceServer
	engine    engineClient
	admission *AdmissionController
	metrics   *Metrics
}

func New(engineClient engineClient, maxQueue int) *Server {
	return &Server{
		engine:    engineClient,
		admission: NewAdmissionController(maxQueue),
		metrics:   NewMetrics(),
	}
}

func (s *Server) Metrics() *Metrics {
	return s.metrics
}

// mapFinishReason translates the engine's FinishReason to the gateway's.
// The two enums are NOT numerically aligned (engine has no STOP value,
// since stop-sequence matching happens in the engine's tokenizer-agnostic
// mock only when EOS is emitted) — a bare int32 cast would silently mislabel
// EOS-finished sequences as FINISH_REASON_STOP (engine EOS=1 == gateway
// STOP=1). This mapping is the fix.
func mapFinishReason(r enginepb.FinishReason) gatewaypb.FinishReason {
	switch r {
	case enginepb.FinishReason_FINISH_REASON_EOS:
		return gatewaypb.FinishReason_FINISH_REASON_EOS
	case enginepb.FinishReason_FINISH_REASON_LENGTH:
		return gatewaypb.FinishReason_FINISH_REASON_LENGTH
	default:
		return gatewaypb.FinishReason_FINISH_REASON_NONE
	}
}

func (s *Server) Generate(
	req *gatewaypb.GenerateRequest,
	stream gatewaypb.InferenceService_GenerateServer,
) error {
	// Admission control
	active, ok := s.admission.TryAcquire()
	if !ok {
		s.metrics.RejectedTotal.Inc()
		return status.Error(codes.ResourceExhausted,
			"server at capacity — retry later")
	}
	defer s.admission.Release()

	s.metrics.ActiveRequests.Set(float64(active))
	s.metrics.RequestsTotal.Inc()

	requestID := uuid.New().String()
	timer := s.metrics.NewRequestTimer()
	ttftMs := -1.0 // per-request TTFT, tracked locally (not shared state)

	// Submit to C++ engine
	engineStream, err := s.engine.Submit(stream.Context(), &enginepb.EngineRequest{
		RequestId:    requestID,
		Prompt:       req.Prompt,
		MaxNewTokens: req.MaxTokens,
		Temperature:  req.Temperature,
	})
	if err != nil {
		return err
	}

	// Stream tokens back to client
	var completionTokens int32
	for {
		token, err := engineStream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		completionTokens++

		if ttftMs < 0 {
			ttftMs = timer.ElapsedMs()
			s.metrics.TTFT.Observe(ttftMs)
		}

		resp := &gatewaypb.GenerateResponse{
			Token:      token.Text,
			IsFinished: token.IsFinished,
		}

		if token.IsFinished {
			elapsed := timer.ElapsedMs()
			resp.FinishReason = mapFinishReason(token.FinishReason)
			resp.Usage = &gatewaypb.UsageStats{
				CompletionTokens:   completionTokens,
				TotalTimeMs:        float32(elapsed),
				TimeToFirstTokenMs: float32(ttftMs),
			}
		}

		if err := stream.Send(resp); err != nil {
			return err // client disconnected
		}

		if token.IsFinished {
			break
		}
	}

	s.metrics.RequestDuration.Observe(timer.ElapsedMs())
	s.metrics.TokensGenerated.Add(float64(completionTokens))
	return nil
}

func (s *Server) Health(
	ctx context.Context,
	req *gatewaypb.HealthRequest,
) (*gatewaypb.HealthResponse, error) {
	// Pull metrics from engine to report queue depth
	engineMetrics, err := s.engine.GetMetrics(ctx)
	if err != nil {
		return &gatewaypb.HealthResponse{Ready: false}, nil
	}
	s.metrics.QueueDepth.Set(float64(engineMetrics.QueueDepth))
	s.metrics.BatchOccupancy.Set(float64(engineMetrics.BatchOccupancyPct))
	return &gatewaypb.HealthResponse{
		Ready:          true,
		QueueDepth:     engineMetrics.QueueDepth,
		ActiveRequests: int32(s.admission.Active()),
	}, nil
}

// PollEngineMetrics periodically pulls engine metrics (queue depth, batch
// occupancy) and republishes them as gateway gauges, so /metrics reflects
// live engine state without requiring an in-flight Health() call. Blocks
// until ctx is done — run it in its own goroutine.
func (s *Server) PollEngineMetrics(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			m, err := s.engine.GetMetrics(ctx)
			if err != nil {
				continue
			}
			s.metrics.QueueDepth.Set(float64(m.QueueDepth))
			s.metrics.BatchOccupancy.Set(float64(m.BatchOccupancyPct))
		}
	}
}
