// Package engine wraps the gRPC connection to the C++ inference engine.
package engine

import (
	"context"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	enginepb "llm-server/gen/engine"
)

// Client connects to the C++ engine, normally over a local Unix domain
// socket. Inference steps take milliseconds; gRPC serialization takes
// microseconds — the boundary overhead is unmeasurable.
type Client struct {
	conn   *grpc.ClientConn
	engine enginepb.InferenceEngineClient
}

func NewClient(addr string) (*Client, error) {
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}
	return &Client{conn: conn, engine: enginepb.NewInferenceEngineClient(conn)}, nil
}

func (c *Client) Submit(ctx context.Context, req *enginepb.EngineRequest) (enginepb.InferenceEngine_SubmitRequestClient, error) {
	return c.engine.SubmitRequest(ctx, req)
}

func (c *Client) GetMetrics(ctx context.Context) (*enginepb.EngineMetrics, error) {
	return c.engine.GetMetrics(ctx, &enginepb.MetricsRequest{})
}

func (c *Client) Close() error {
	return c.conn.Close()
}
