package server

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Metrics holds the 8+ Prometheus collectors the resume bullet promises.
// Each Metrics owns a private registry rather than using the global
// prometheus.DefaultRegisterer: that lets tests construct multiple
// independent Servers (and therefore multiple Metrics) in the same process
// without a "duplicate metrics collector registration" panic, and it keeps
// main.go's /metrics endpoint free of Go runtime metrics noise unless we
// choose to add it.
type Metrics struct {
	Registry *prometheus.Registry

	// 1. Total requests received
	RequestsTotal prometheus.Counter
	// 2. Total requests rejected (queue full)
	RejectedTotal prometheus.Counter
	// 3. Currently active requests
	ActiveRequests prometheus.Gauge
	// 4. Request duration (end-to-end latency histogram, ms)
	RequestDuration prometheus.Histogram
	// 5. Time to first token (histogram, ms)
	TTFT prometheus.Histogram
	// 6. Total tokens generated
	TokensGenerated prometheus.Counter
	// 7. Batch occupancy (from engine, updated on health checks)
	BatchOccupancy prometheus.Gauge
	// 8. Queue depth (from engine)
	QueueDepth prometheus.Gauge
}

func NewMetrics() *Metrics {
	reg := prometheus.NewRegistry()
	f := promauto.With(reg)

	return &Metrics{
		Registry: reg,
		RequestsTotal: f.NewCounter(prometheus.CounterOpts{
			Name: "inference_requests_total",
			Help: "Total inference requests received",
		}),
		RejectedTotal: f.NewCounter(prometheus.CounterOpts{
			Name: "inference_requests_rejected_total",
			Help: "Requests rejected due to queue full",
		}),
		ActiveRequests: f.NewGauge(prometheus.GaugeOpts{
			Name: "inference_requests_active",
			Help: "Currently active requests",
		}),
		RequestDuration: f.NewHistogram(prometheus.HistogramOpts{
			Name:    "inference_request_duration_ms",
			Help:    "End-to-end request duration in milliseconds",
			Buckets: prometheus.ExponentialBuckets(10, 2, 12), // 10ms to 40s
		}),
		TTFT: f.NewHistogram(prometheus.HistogramOpts{
			Name:    "inference_ttft_ms",
			Help:    "Time to first token in milliseconds",
			Buckets: prometheus.ExponentialBuckets(5, 2, 10), // 5ms to 5s
		}),
		TokensGenerated: f.NewCounter(prometheus.CounterOpts{
			Name: "inference_tokens_generated_total",
			Help: "Total tokens generated",
		}),
		BatchOccupancy: f.NewGauge(prometheus.GaugeOpts{
			Name: "inference_batch_occupancy_pct",
			Help: "Current batch occupancy percentage",
		}),
		QueueDepth: f.NewGauge(prometheus.GaugeOpts{
			Name: "inference_queue_depth",
			Help: "Current request queue depth",
		}),
	}
}

type RequestTimer struct {
	start time.Time
}

func (m *Metrics) NewRequestTimer() *RequestTimer {
	return &RequestTimer{start: time.Now()}
}

func (t *RequestTimer) ElapsedMs() float64 {
	return float64(time.Since(t.start).Nanoseconds()) / 1e6
}
