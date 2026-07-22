#include "grpc_service.h"

EngineServiceImpl::EngineServiceImpl(ContinuousBatchScheduler* scheduler)
    : scheduler_(scheduler) {
    // Register our callback with the scheduler
    scheduler_->set_token_callback(
        [this](const std::string& rid, const std::string& text,
               int32_t tid, bool done, int reason) {
            on_token(rid, text, tid, done, reason);
        });
}

grpc::Status EngineServiceImpl::SubmitRequest(
    grpc::ServerContext* context,
    const engine::EngineRequest* request,
    grpc::ServerWriter<engine::EngineToken>* writer) {

    // Create per-request stream
    auto stream = std::make_shared<RequestStream>();
    {
        std::lock_guard<std::mutex> lock(streams_mutex_);
        streams_[request->request_id()] = stream;
    }

    // Submit to scheduler
    bool accepted = scheduler_->submit(
        request->request_id(),
        request->prompt(),
        request->max_new_tokens(),
        request->temperature());

    if (!accepted) {
        std::lock_guard<std::mutex> lock(streams_mutex_);
        streams_.erase(request->request_id());
        return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED,
                           "Queue full");
    }

    // Stream tokens until done
    while (true) {
        engine::EngineToken token;
        {
            std::unique_lock<std::mutex> lock(stream->mu);
            stream->cv.wait(lock, [&] {
                return !stream->tokens.empty() || stream->done;
            });

            if (stream->tokens.empty() && stream->done) break;

            token = std::move(stream->tokens.front());
            stream->tokens.pop();
        }

        if (!writer->Write(token)) break;  // client disconnected
        if (token.is_finished()) break;
    }

    // Cleanup
    {
        std::lock_guard<std::mutex> lock(streams_mutex_);
        streams_.erase(request->request_id());
    }

    return grpc::Status::OK;
}

void EngineServiceImpl::on_token(
    const std::string& request_id,
    const std::string& text,
    int32_t token_id,
    bool is_finished,
    int finish_reason) {

    std::shared_ptr<RequestStream> stream;
    {
        std::lock_guard<std::mutex> lock(streams_mutex_);
        auto it = streams_.find(request_id);
        if (it == streams_.end()) return;  // client already disconnected
        stream = it->second;
    }

    engine::EngineToken token;
    token.set_request_id(request_id);
    token.set_token_id(token_id);
    token.set_text(text);
    token.set_is_finished(is_finished);
    token.set_finish_reason(
        static_cast<engine::FinishReason>(finish_reason));

    {
        std::lock_guard<std::mutex> lock(stream->mu);
        stream->tokens.push(std::move(token));
        if (is_finished) stream->done = true;
    }
    stream->cv.notify_one();
}

grpc::Status EngineServiceImpl::GetMetrics(
    grpc::ServerContext* /*context*/,
    const engine::MetricsRequest* /*request*/,
    engine::EngineMetrics* response) {

    const auto& m = scheduler_->metrics();
    response->set_total_tokens_generated(m.total_tokens_generated.load());
    response->set_current_batch_size(m.current_batch_size.load());
    response->set_max_batch_size(scheduler_->max_batch_size());
    response->set_queue_depth(m.queue_depth.load());
    response->set_batch_occupancy_pct(m.batch_occupancy_pct.load());
    response->set_tokens_per_second(m.tokens_per_second.load());
    response->set_kv_cache_slots_used(m.kv_slots_used.load());
    response->set_kv_cache_slots_total(scheduler_->max_batch_size());

    return grpc::Status::OK;
}
