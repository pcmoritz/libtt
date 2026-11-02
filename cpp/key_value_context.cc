#include "cpp/key_value_context.h"

#include "tt_metal/distributed/multihost/single_host_context.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <future>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace libtt {
namespace {

namespace mh = tt::tt_metal::distributed::multihost;
using Bytes = ttsl::Span<std::byte>;

void Check(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

std::string_view AsString(Bytes bytes) {
  return {reinterpret_cast<const char *>(bytes.data()), bytes.size()};
}

void Copy(std::string_view value, Bytes buffer) {
  Check(value.size() == buffer.size(), "Coordination message size mismatch");
  if (!buffer.empty())
    std::memcpy(buffer.data(), value.data(), buffer.size());
}

mh::ContextPtr MakeContext(KeyValueStore store, std::vector<int> members,
                           int globalRank, std::string prefix);

class KeyValueRequest final : public mh::Request {
public:
  explicit KeyValueRequest(std::future<mh::Status> result)
      : result_(result.share()) {}

  mh::Status wait() override { return result_.get(); }
  std::optional<mh::Status> test() override {
    return active() ? std::nullopt : std::optional{result_.get()};
  }
  void cancel() override {
    throw std::runtime_error(
        "Coordination request cancellation is unsupported");
  }
  bool active() const override {
    return result_.wait_for(std::chrono::seconds(0)) !=
           std::future_status::ready;
  }

private:
  std::shared_future<mh::Status> result_;
};

// Reuse SingleHostContext's explicit errors for unsupported host operations.
// In particular, this does not emulate device reductions through the store.
class KeyValueContext final : public mh::SingleHostContext {
public:
  KeyValueContext(KeyValueStore store, std::vector<int> members, int globalRank,
                  std::string prefix)
      : store_(std::move(store)), members_(std::move(members)),
        globalRank_(globalRank), prefix_(std::move(prefix)) {
    const auto it = std::find(members_.begin(), members_.end(), globalRank_);
    Check(it != members_.end(), "Process is not a member of this context");
    rank_ = static_cast<int>(it - members_.begin());
  }

  mh::Rank rank() const override { return mh::Rank{rank_}; }
  mh::Size size() const override {
    return mh::Size{static_cast<int>(members_.size())};
  }

  void send(Bytes buffer, mh::Rank dest, mh::Tag tag) const override {
    const auto key = MessageKey(rank(), dest, tag, true);
    store_.put(key, AsString(buffer));
  }

  void ssend(Bytes buffer, mh::Rank dest, mh::Tag tag) const override {
    const auto key = MessageKey(rank(), dest, tag, true);
    store_.put(key, AsString(buffer));
    store_.get(key + "/ack");
  }

  void recv(Bytes buffer, mh::Rank source, mh::Tag tag) const override {
    const auto key = MessageKey(source, rank(), tag, false);
    Copy(store_.get(key), buffer);
    store_.put(key + "/ack", "1");
  }

  mh::RequestPtr irecv(Bytes buffer, mh::Rank source,
                       mh::Tag tag) const override {
    Check(buffer.size() <= std::numeric_limits<int>::max(),
          "Coordination message too large");
    const auto key = MessageKey(source, rank(), tag, false);
    return std::make_shared<KeyValueRequest>(std::async(
        std::launch::async, [store = store_, key, buffer, source, tag] {
          Copy(store.get(key), buffer);
          store.put(key + "/ack", "1");
          return mh::Status{source, tag, static_cast<int>(buffer.size())};
        }));
  }

  std::size_t snoop_incoming_msg_size(mh::Rank source,
                                      mh::Tag tag) const override {
    return store_.get(MessageKey(source, rank(), tag, false, false)).size();
  }

  void all_gather(Bytes send, Bytes recv) const override {
    Check(recv.size() == send.size() * members_.size(),
          "Invalid all_gather receive buffer size");
    if (members_.size() == 1) {
      Copy(AsString(send), recv);
      return;
    }
    const auto key = Next("collective") + "/all_gather/";
    store_.put(key + std::to_string(rank_), AsString(send));
    for (int i = 0; i < *size(); ++i)
      Copy(store_.get(key + std::to_string(i)),
           recv.subspan(i * send.size(), send.size()));
  }

  void barrier() const override {
    std::byte value{1};
    std::vector<std::byte> values(members_.size());
    all_gather({&value, 1}, values);
  }

  void broadcast(Bytes buffer, mh::Rank root) const override {
    ValidateRank(root);
    if (members_.size() == 1)
      return;
    const auto key = Next("collective") + "/broadcast/" + std::to_string(*root);
    if (rank() == root)
      store_.put(key, AsString(buffer));
    else
      Copy(store_.get(key), buffer);
  }

  mh::ContextPtr create_sub_context(ttsl::Span<int> ranks) const override {
    std::vector<int> members;
    std::string group = "group";
    for (int r : ranks) {
      ValidateRank(mh::Rank{r});
      Check(std::find(members.begin(), members.end(), members_[r]) ==
                members.end(),
            "Duplicate rank in subgroup");
      members.push_back(members_[r]);
      group += "/" + std::to_string(r);
    }
    if (std::find(members.begin(), members.end(), globalRank_) == members.end())
      return nullptr;
    return MakeContext(store_, std::move(members), globalRank_, Next(group));
  }

  mh::ContextPtr duplicate() const override {
    return MakeContext(store_, members_, globalRank_, Next("duplicate"));
  }

  mh::ContextPtr split(mh::Color color, mh::Key key) const override {
    std::array<int, 2> local{*color, *key};
    std::vector<std::array<int, 2>> values(members_.size());
    all_gather(ttsl::as_writable_bytes(ttsl::Span<int>(local)),
               ttsl::as_writable_bytes(ttsl::Span<std::array<int, 2>>(values)));
    if (*color == SPLIT_COLOR_UNDEFINED)
      return nullptr;
    std::vector<int> ranks;
    for (int i = 0; i < *size(); ++i)
      if (values[i][0] == *color)
        ranks.push_back(i);
    std::stable_sort(ranks.begin(), ranks.end(),
                     [&](int a, int b) { return values[a][1] < values[b][1]; });
    return create_sub_context(ranks);
  }

  void translate_ranks_to_other_ctx(ttsl::Span<int> ranks,
                                    const mh::ContextPtr &other,
                                    ttsl::Span<int> translated) const override {
    const auto target = std::dynamic_pointer_cast<KeyValueContext>(other);
    Check(target != nullptr && ranks.size() == translated.size(),
          "Invalid context rank translation");
    for (std::size_t i = 0; i < ranks.size(); ++i) {
      ValidateRank(mh::Rank{ranks[i]});
      const auto it = std::find(target->members_.begin(),
                                target->members_.end(), members_[ranks[i]]);
      translated[i] = it == target->members_.end()
                          ? -1
                          : static_cast<int>(it - target->members_.begin());
    }
  }

private:
  void ValidateRank(mh::Rank rank) const {
    Check(*rank >= 0 && *rank < *size(), "Coordination rank out of range");
  }

  std::string Next(const std::string &channel) const {
    std::lock_guard lock(mutex_);
    return prefix_ + "/" + channel + "/" +
           std::to_string(sequences_[channel]++);
  }

  std::string MessageKey(mh::Rank src, mh::Rank dst, mh::Tag tag, bool sending,
                         bool advance = true) const {
    ValidateRank(src);
    ValidateRank(dst);
    const auto channel = "message/" + std::to_string(*src) + "/" +
                         std::to_string(*dst) + "/" + std::to_string(*tag);
    std::lock_guard lock(mutex_);
    auto &sequence = sequences_[(sending ? "send/" : "recv/") + channel];
    const auto key = prefix_ + "/" + channel + "/" + std::to_string(sequence);
    if (advance)
      ++sequence;
    return key;
  }

  KeyValueStore store_;
  std::vector<int> members_;
  int globalRank_;
  int rank_;
  std::string prefix_;
  mutable std::mutex mutex_;
  mutable std::map<std::string, std::size_t> sequences_;
};

mh::ContextPtr MakeContext(KeyValueStore store, std::vector<int> members,
                           int globalRank, std::string prefix) {
  // Upstream's context ID allocator is not thread-safe.
  static std::mutex creationMutex;
  std::lock_guard lock(creationMutex);
  return std::make_shared<KeyValueContext>(std::move(store), std::move(members),
                                           globalRank, std::move(prefix));
}

} // namespace

mh::ContextPtr MakeKeyValueContext(KeyValueStore store, int rank, int size) {
  Check(size > 0 && rank >= 0 && rank < size,
        "Invalid distributed process configuration");
  Check(store.put && store.get,
        "Distributed execution requires a key-value store");
  std::vector<int> members(size);
  std::iota(members.begin(), members.end(), 0);
  return MakeContext(std::move(store), std::move(members), rank,
                     "libtt/control");
}

} // namespace libtt
