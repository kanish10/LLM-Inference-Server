#include "scheduler.h"
#include <algorithm>
#include <cassert>
#include <thread>

ContinuousBatchScheduler::ContinuousBatchScheduler(
    Config config, std::unique_ptr<TokenGenerator> generator)
    : config_(config),
      generator_(std::move(generator)),
      kv_cache_(config.max_batch_size,
                config.max_sequence_length,
                config.num_layers,
                config.num_kv_heads,
                config.head_dim) {
    active_batch_.reserve(config.max_batch_size);
}

bool ContinuousBatchScheduler::submit(
    const std::string& request_id,
    const std::string& prompt,
    int32_t max_new_tokens,
    float temperature) {

    Sequence seq;
    seq.request_id = request_id;
    seq.prompt_tokens = generator_->tokenize(prompt);
    seq.max_new_tokens = max_new_tokens;
    seq.temperature = temperature;
    seq.state = SequenceState::WAITING;
    seq.arrival_time = std::chrono::steady_clock::now();

    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (static_cast<int>(waiting_queue_.size()) >= config_.max_queue_depth) {
        return false;  // queue full — caller returns RESOURCE_EXHAUSTED
    }
    waiting_queue_.push_back(std::move(seq));
    metrics_.total_requests_submitted.fetch_add(1);
    metrics_.queue_depth.store(static_cast<int>(waiting_queue_.size()));
    return true;
}

void ContinuousBatchScheduler::set_token_callback(TokenCallback cb) {
    token_callback_ = std::move(cb);
}

// ─────────────────────────────────────────────
// THE MAIN LOOP
// ─────────────────────────────────────────────
void ContinuousBatchScheduler::run() {
    running_.store(true);
    while (running_.load()) {
        // If batch is empty AND queue is empty, sleep briefly to avoid busy-wait
        if (active_batch_.empty()) {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (waiting_queue_.empty()) {
                lock.unlock();
                // (In production: use a condition variable instead)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
        }

        admit_new();       // fill empty slots from queue
        prefill_step();    // process new sequences' prompts
        decode_step();     // generate one token per DECODE sequence
        check_finished();  // mark sequences that are done
        if (config_.static_batching) {
            evict_finished_static();
        } else {
            evict_finished();  // remove them, free KV slots
        }
        update_metrics();  // rolling rate window
    }
}

void ContinuousBatchScheduler::stop() {
    running_.store(false);
}

// ─────────────────────────────────────────────
// PHASE 1: ADMIT
// Fill empty batch slots from the waiting queue.
// ─────────────────────────────────────────────
void ContinuousBatchScheduler::admit_new() {
    int free_slots = config_.max_batch_size
                     - static_cast<int>(active_batch_.size());
    if (free_slots <= 0) return;

    std::lock_guard<std::mutex> lock(queue_mutex_);
    int to_admit = std::min(free_slots,
                            static_cast<int>(waiting_queue_.size()));

    for (int i = 0; i < to_admit; ++i) {
        Sequence seq = std::move(waiting_queue_.front());
        waiting_queue_.pop_front();

        int slot = kv_cache_.allocate();
        assert(slot >= 0);  // free_slots guarantees availability

        seq.kv_slot = slot;
        seq.state = SequenceState::PREFILL;
        active_batch_.push_back(std::move(seq));
    }

    metrics_.queue_depth.store(static_cast<int>(waiting_queue_.size()));
    metrics_.kv_slots_used.store(
        kv_cache_.num_total() - kv_cache_.num_free());
}

// ─────────────────────────────────────────────
// PHASE 2: PREFILL
// For newly admitted sequences, process the full prompt.
// In real inference: this is a single forward pass with all
// prompt tokens, filling the KV cache. It's compute-bound
// (unlike decode, which is memory-bandwidth-bound).
// ─────────────────────────────────────────────
void ContinuousBatchScheduler::prefill_step() {
    TokenGenerator::BatchInput prefill_batch;

    for (auto& seq : active_batch_) {
        if (seq.state != SequenceState::PREFILL) continue;

        TokenGenerator::Entry entry;
        entry.kv_slot = seq.kv_slot;
        entry.tokens = seq.prompt_tokens;  // all prompt tokens
        entry.seq_position = 0;
        entry.temperature = seq.temperature;
        prefill_batch.entries.push_back(entry);
    }

    if (prefill_batch.entries.empty()) return;

    // Forward pass: processes all prompt tokens, fills KV cache,
    // returns the first generated token for each sequence.
    auto output = generator_->forward(prefill_batch);

    // Match outputs back to sequences
    int out_idx = 0;
    for (auto& seq : active_batch_) {
        if (seq.state != SequenceState::PREFILL) continue;

        int32_t token = output.next_tokens[out_idx++];
        seq.generated_tokens.push_back(token);
        seq.state = SequenceState::DECODE;
        seq.first_token_time = std::chrono::steady_clock::now();

        // A sequence can finish on its very first token (EOS immediately,
        // or max_new_tokens == 1) — check both, same as decode_step.
        bool is_eos = (token == generator_->eos_token_id());
        bool is_finished = is_eos || seq.hit_max();

        if (token_callback_) {
            token_callback_(
                seq.request_id,
                generator_->detokenize(token),
                token,
                is_finished,
                is_eos ? 1 : (is_finished ? 2 : 0)
            );
        }

        if (is_finished) {
            seq.state = SequenceState::FINISHED;
            seq.completion_time = std::chrono::steady_clock::now();
        }

        metrics_.total_tokens_generated.fetch_add(1);
    }
}

// ─────────────────────────────────────────────
// PHASE 3: DECODE
// Generate one token for every active DECODE sequence.
// This is the step that runs most often — the "inner loop."
// ─────────────────────────────────────────────
void ContinuousBatchScheduler::decode_step() {
    TokenGenerator::BatchInput decode_batch;
    std::vector<int> batch_indices;  // map from decode_batch index to active_batch_ index

    for (int i = 0; i < static_cast<int>(active_batch_.size()); ++i) {
        auto& seq = active_batch_[i];
        if (seq.state != SequenceState::DECODE) continue;

        TokenGenerator::Entry entry;
        entry.kv_slot = seq.kv_slot;
        entry.tokens = {seq.generated_tokens.back()};  // just the last token
        entry.seq_position = seq.total_length() - 1;
        entry.temperature = seq.temperature;
        decode_batch.entries.push_back(entry);
        batch_indices.push_back(i);
    }

    if (decode_batch.entries.empty()) return;

    auto output = generator_->forward(decode_batch);

    for (int j = 0; j < static_cast<int>(batch_indices.size()); ++j) {
        auto& seq = active_batch_[batch_indices[j]];
        int32_t token = output.next_tokens[j];
        seq.generated_tokens.push_back(token);

        bool is_eos = (token == generator_->eos_token_id());
        bool is_finished = is_eos || seq.hit_max();

        if (token_callback_) {
            token_callback_(
                seq.request_id,
                generator_->detokenize(token),
                token,
                is_finished,
                is_eos ? 1 : (is_finished ? 2 : 0)
            );
        }

        metrics_.total_tokens_generated.fetch_add(1);
    }
}

// ─────────────────────────────────────────────
// PHASE 4: CHECK
// Mark sequences as FINISHED if they hit EOS or max tokens.
// Separated from eviction for clarity and testability.
// ─────────────────────────────────────────────
void ContinuousBatchScheduler::check_finished() {
    for (auto& seq : active_batch_) {
        if (seq.state != SequenceState::DECODE) continue;

        bool hit_eos = (!seq.generated_tokens.empty() &&
                        seq.generated_tokens.back() == generator_->eos_token_id());
        bool hit_max = seq.hit_max();

        if (hit_eos || hit_max) {
            seq.state = SequenceState::FINISHED;
            seq.completion_time = std::chrono::steady_clock::now();
        }
    }
}

// ─────────────────────────────────────────────
// PHASE 5: EVICT (continuous batching)
// Remove finished sequences. Free their KV cache slots.
// THIS IS THE LINE THAT MAKES CONTINUOUS BATCHING WORK.
// In static batching, this phase doesn't exist — you wait
// until ALL sequences are done, then free everything at once.
// ─────────────────────────────────────────────
void ContinuousBatchScheduler::evict_finished() {
    auto it = std::remove_if(
        active_batch_.begin(), active_batch_.end(),
        [this](Sequence& seq) {
            if (seq.state == SequenceState::FINISHED) {
                kv_cache_.free(seq.kv_slot);  // <-- THE KEY OPERATION
                generator_->reset_sequence(seq.kv_slot);
                metrics_.total_requests_completed.fetch_add(1);
                return true;
            }
            return false;
        });
    active_batch_.erase(it, active_batch_.end());

    metrics_.current_batch_size.store(
        static_cast<int>(active_batch_.size()));
    metrics_.kv_slots_used.store(
        kv_cache_.num_total() - kv_cache_.num_free());
}

// ─────────────────────────────────────────────
// PHASE 5b: EVICT (static batching baseline)
// Wait until every sequence in the batch is FINISHED, then free
// them all at once. This is the ONLY difference between static and
// continuous batching — see CLAUDE.md "Static Batching Baseline".
// ─────────────────────────────────────────────
void ContinuousBatchScheduler::evict_finished_static() {
    bool all_done = !active_batch_.empty() && std::all_of(
        active_batch_.begin(), active_batch_.end(),
        [](const Sequence& s) { return s.state == SequenceState::FINISHED; });

    if (all_done) {
        for (auto& seq : active_batch_) {
            kv_cache_.free(seq.kv_slot);
            generator_->reset_sequence(seq.kv_slot);
            metrics_.total_requests_completed.fetch_add(1);
        }
        active_batch_.clear();
    }

    // Unlike the early-return version, always refresh the gauges: a
    // finished-but-not-evicted batch (the whole point of static batching)
    // must still report its true occupied size, not go stale at whatever
    // it was after the last eviction.
    metrics_.current_batch_size.store(
        static_cast<int>(active_batch_.size()));
    metrics_.kv_slots_used.store(
        kv_cache_.num_total() - kv_cache_.num_free());
}

// ─────────────────────────────────────────────
// PHASE 6: METRICS
// ─────────────────────────────────────────────
void ContinuousBatchScheduler::update_metrics() {
    auto now = std::chrono::steady_clock::now();
    int bs = static_cast<int>(active_batch_.size());

    // Rolling batch occupancy
    float occ = static_cast<float>(bs) / config_.max_batch_size;
    float prev = metrics_.batch_occupancy_pct.load();
    float alpha = 0.05f;  // EMA smoothing
    metrics_.batch_occupancy_pct.store(prev * (1 - alpha) + occ * alpha);

    // Tokens/second (windowed)
    rate_window_.record(bs, now);  // bs tokens generated this step
    metrics_.tokens_per_second.store(rate_window_.compute_rate());
}
