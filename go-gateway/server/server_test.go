package server

import (
	"context"
	"io"
	"net"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"

	enginepb "llm-server/gen/engine"
	gatewaypb "llm-server/gen/gateway"
)

// fakeEngine implements enginepb.InferenceEngineServer, standing in for the
// C++ engine process so the gateway's admission/streaming/metrics logic can
// be tested without building and running the C++ binary.
type fakeEngine struct {
	enginepb.UnimplementedInferenceEngineServer
	metrics *enginepb.EngineMetrics
	// if rejectSubmit is true, SubmitRequest fails immediately with
	// RESOURCE_EXHAUSTED, simulating the engine's own queue-full response.
	rejectSubmit bool
}

func (f *fakeEngine) SubmitRequest(req *enginepb.EngineRequest, stream enginepb.InferenceEngine_SubmitRequestServer) error {
	if f.rejectSubmit {
		return status.Error(codes.ResourceExhausted, "engine queue full")
	}
	for i := int32(0); i < req.MaxNewTokens; i++ {
		finished := i == req.MaxNewTokens-1
		reason := enginepb.FinishReason_FINISH_REASON_NONE
		if finished {
			reason = enginepb.FinishReason_FINISH_REASON_LENGTH
		}
		if err := stream.Send(&enginepb.EngineToken{
			RequestId:    req.RequestId,
			TokenId:      i,
			Text:         "x",
			IsFinished:   finished,
			FinishReason: reason,
		}); err != nil {
			return err
		}
	}
	return nil
}

func (f *fakeEngine) GetMetrics(ctx context.Context, req *enginepb.MetricsRequest) (*enginepb.EngineMetrics, error) {
	if f.metrics != nil {
		return f.metrics, nil
	}
	return &enginepb.EngineMetrics{}, nil
}

// grpcEngineClient adapts a raw enginepb.InferenceEngineClient to the
// engineClient interface server.go depends on — the same shape
// engine.Client (go-gateway/engine) implements against a real process.
type grpcEngineClient struct {
	c enginepb.InferenceEngineClient
}

func (g *grpcEngineClient) Submit(ctx context.Context, req *enginepb.EngineRequest) (enginepb.InferenceEngine_SubmitRequestClient, error) {
	return g.c.SubmitRequest(ctx, req)
}

func (g *grpcEngineClient) GetMetrics(ctx context.Context) (*enginepb.EngineMetrics, error) {
	return g.c.GetMetrics(ctx, &enginepb.MetricsRequest{})
}

// testHarness wires: gatewaypb client -> (bufconn) -> Server -> engineClient
// -> (bufconn) -> fakeEngine. Both hops are real gRPC over in-memory
// connections, so this exercises the actual streaming/serialization path.
type testHarness struct {
	gatewayClient gatewaypb.InferenceServiceClient
	srv           *Server
	closeFn       func()
}

func newTestHarness(t *testing.T, fe *fakeEngine, maxQueue int) *testHarness {
	t.Helper()

	// Engine-side bufconn server
	engineLis := bufconn.Listen(1024 * 1024)
	engineGrpc := grpc.NewServer()
	enginepb.RegisterInferenceEngineServer(engineGrpc, fe)
	go func() { _ = engineGrpc.Serve(engineLis) }()

	engineConn, err := grpc.NewClient("passthrough:///bufnet",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) {
			return engineLis.DialContext(ctx)
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		t.Fatalf("dial engine bufconn: %v", err)
	}

	srv := New(&grpcEngineClient{c: enginepb.NewInferenceEngineClient(engineConn)}, maxQueue)

	// Gateway-side bufconn server, fronting srv exactly like main.go does
	gwLis := bufconn.Listen(1024 * 1024)
	gwGrpc := grpc.NewServer()
	gatewaypb.RegisterInferenceServiceServer(gwGrpc, srv)
	go func() { _ = gwGrpc.Serve(gwLis) }()

	gwConn, err := grpc.NewClient("passthrough:///bufnet",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) {
			return gwLis.DialContext(ctx)
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		t.Fatalf("dial gateway bufconn: %v", err)
	}

	return &testHarness{
		gatewayClient: gatewaypb.NewInferenceServiceClient(gwConn),
		srv:           srv,
		closeFn: func() {
			_ = gwConn.Close()
			_ = engineConn.Close()
			gwGrpc.Stop()
			engineGrpc.Stop()
		},
	}
}

func TestGenerateStreamsAllTokensAndUsage(t *testing.T) {
	h := newTestHarness(t, &fakeEngine{}, 64)
	defer h.closeFn()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream, err := h.gatewayClient.Generate(ctx, &gatewaypb.GenerateRequest{
		Prompt:    "hello",
		MaxTokens: 5,
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}

	var responses []*gatewaypb.GenerateResponse
	for {
		resp, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
		responses = append(responses, resp)
	}

	if len(responses) != 5 {
		t.Fatalf("got %d responses, want 5", len(responses))
	}
	last := responses[len(responses)-1]
	if !last.IsFinished {
		t.Fatal("last response should have IsFinished=true")
	}
	if last.FinishReason != gatewaypb.FinishReason_FINISH_REASON_LENGTH {
		t.Fatalf("FinishReason = %v, want FINISH_REASON_LENGTH", last.FinishReason)
	}
	if last.Usage == nil {
		t.Fatal("final response must carry Usage")
	}
	if last.Usage.CompletionTokens != 5 {
		t.Fatalf("CompletionTokens = %d, want 5", last.Usage.CompletionTokens)
	}
	if last.Usage.TimeToFirstTokenMs < 0 {
		t.Fatalf("TimeToFirstTokenMs = %v, want >= 0", last.Usage.TimeToFirstTokenMs)
	}
	for i, r := range responses[:len(responses)-1] {
		if r.IsFinished {
			t.Fatalf("response %d finished early", i)
		}
	}
}

func TestGenerateRejectsWhenAtCapacity(t *testing.T) {
	// maxQueue=0: the very first request already exceeds capacity.
	h := newTestHarness(t, &fakeEngine{}, 0)
	defer h.closeFn()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream, err := h.gatewayClient.Generate(ctx, &gatewaypb.GenerateRequest{
		Prompt:    "hello",
		MaxTokens: 5,
	})
	if err != nil {
		t.Fatalf("Generate call itself should not error: %v", err)
	}

	_, err = stream.Recv()
	if err == nil {
		t.Fatal("expected ResourceExhausted error, got nil")
	}
	st, ok := status.FromError(err)
	if !ok || st.Code() != codes.ResourceExhausted {
		t.Fatalf("expected ResourceExhausted, got %v", err)
	}
}

func TestGenerateReturnsErrorWhenEngineRejects(t *testing.T) {
	// Simulates the engine's own waiting_queue_ being full: the gateway's
	// admission slot is granted, but the engine rejects on SubmitRequest.
	// The gateway must surface that failure rather than swallow it.
	h := newTestHarness(t, &fakeEngine{rejectSubmit: true}, 64)
	defer h.closeFn()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream, err := h.gatewayClient.Generate(ctx, &gatewaypb.GenerateRequest{
		Prompt:    "hello",
		MaxTokens: 5,
	})
	if err != nil {
		t.Fatalf("Generate call itself should not error: %v", err)
	}

	_, err = stream.Recv()
	if err == nil {
		t.Fatal("expected error from engine rejection, got nil")
	}
	st, ok := status.FromError(err)
	if !ok || st.Code() != codes.ResourceExhausted {
		t.Fatalf("expected ResourceExhausted propagated from engine, got %v", err)
	}
}

func TestHealthReportsEngineMetrics(t *testing.T) {
	fe := &fakeEngine{metrics: &enginepb.EngineMetrics{
		QueueDepth:        3,
		BatchOccupancyPct: 0.75,
	}}
	h := newTestHarness(t, fe, 64)
	defer h.closeFn()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resp, err := h.gatewayClient.Health(ctx, &gatewaypb.HealthRequest{})
	if err != nil {
		t.Fatalf("Health: %v", err)
	}
	if !resp.Ready {
		t.Fatal("expected Ready=true")
	}
	if resp.QueueDepth != 3 {
		t.Fatalf("QueueDepth = %d, want 3", resp.QueueDepth)
	}
}

func TestMapFinishReason(t *testing.T) {
	cases := []struct {
		in   enginepb.FinishReason
		want gatewaypb.FinishReason
	}{
		{enginepb.FinishReason_FINISH_REASON_NONE, gatewaypb.FinishReason_FINISH_REASON_NONE},
		{enginepb.FinishReason_FINISH_REASON_EOS, gatewaypb.FinishReason_FINISH_REASON_EOS},
		{enginepb.FinishReason_FINISH_REASON_LENGTH, gatewaypb.FinishReason_FINISH_REASON_LENGTH},
	}
	for _, c := range cases {
		if got := mapFinishReason(c.in); got != c.want {
			t.Errorf("mapFinishReason(%v) = %v, want %v", c.in, got, c.want)
		}
	}
}
