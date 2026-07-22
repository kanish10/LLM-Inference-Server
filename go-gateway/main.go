package main

import (
	"context"
	"flag"
	"log"
	"net"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"google.golang.org/grpc"

	"llm-server/engine"
	gatewaypb "llm-server/gen/gateway"
	"llm-server/server"
)

func main() {
	listenAddr := flag.String("listen", ":50051", "gRPC listen address")
	metricsAddr := flag.String("metrics", ":9090", "Prometheus metrics address")
	engineAddr := flag.String("engine", "unix:///tmp/inference-engine.sock", "C++ engine address")
	maxQueue := flag.Int("max-queue", 64, "Max active requests before rejecting")
	flag.Parse()

	// Connect to C++ engine
	engineClient, err := engine.NewClient(*engineAddr)
	if err != nil {
		log.Fatalf("Failed to connect to engine: %v", err)
	}
	defer engineClient.Close()

	// Create server
	srv := server.New(engineClient, *maxQueue)

	pollCtx, cancelPoll := context.WithCancel(context.Background())
	defer cancelPoll()
	go srv.PollEngineMetrics(pollCtx, 2*time.Second)

	// Start Prometheus metrics endpoint
	go func() {
		mux := http.NewServeMux()
		mux.Handle("/metrics", promhttp.HandlerFor(srv.Metrics().Registry, promhttp.HandlerOpts{}))
		log.Printf("Metrics on %s/metrics", *metricsAddr)
		log.Fatal(http.ListenAndServe(*metricsAddr, mux))
	}()

	// Start gRPC server
	lis, err := net.Listen("tcp", *listenAddr)
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}

	grpcServer := grpc.NewServer()
	gatewaypb.RegisterInferenceServiceServer(grpcServer, srv)

	log.Printf("Gateway listening on %s", *listenAddr)
	log.Fatal(grpcServer.Serve(lis))
}
