#include "cpp/key_value_context.h"

#include <array>
#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace {
namespace mh = tt::tt_metal::distributed::multihost;

void Check(bool condition) {
  if (!condition)
    throw std::runtime_error("Coordination test failed");
}

struct Store {
  std::mutex mutex;
  std::condition_variable ready;
  std::map<std::string, std::string> values;

  libtt::KeyValueStore callbacks() {
    return {.put =
                [&](std::string_view key, std::string_view value) {
                  std::lock_guard lock(mutex);
                  Check(values.emplace(key, value).second);
                  ready.notify_all();
                },
            .get =
                [&](std::string_view key) {
                  std::unique_lock lock(mutex);
                  std::string name(key);
                  Check(ready.wait_for(lock, std::chrono::seconds(5),
                                       [&] { return values.contains(name); }));
                  return values.at(name);
                }};
  }
};

void RunRank(const mh::ContextPtr &context) {
  const int rank = *context->rank();
  Check(*context->size() == 2);
  for (int round = 0; round < 3; ++round) {
    int value = 10 * round + rank;
    std::array<int, 2> result{};
    context->all_gather(ttsl::as_writable_bytes(ttsl::Span<int>(&value, 1)),
                        ttsl::as_writable_bytes(ttsl::Span<int>(result)));
    Check(result[0] == 10 * round && result[1] == 10 * round + 1);
    context->barrier();
  }

  // Same message tag, different lengths, and snooping must not consume data.
  for (int length : {1, 7, 3}) {
    std::vector<std::byte> message(length, std::byte{0x35});
    if (rank == 0) {
      context->ssend(message, mh::Rank{1}, mh::Tag{9});
    } else {
      Check(context->snoop_incoming_msg_size(mh::Rank{0}, mh::Tag{9}) ==
            length);
      std::vector<std::byte> received(length);
      context->recv(received, mh::Rank{0}, mh::Tag{9});
      Check(received == message);
    }
  }

  // Posting receives must be nonblocking and preserve same-tag ordering.
  std::array<int, 2> asyncValues{};
  std::array<mh::RequestPtr, 2> requests;
  if (rank == 1) {
    for (int i = 0; i < 2; ++i) {
      requests[i] = context->irecv(
          ttsl::as_writable_bytes(ttsl::Span<int>(&asyncValues[i], 1)),
          mh::Rank{0}, mh::Tag{17});
      Check(requests[i]->active() && !requests[i]->test().has_value());
    }
  }
  context->barrier();
  if (rank == 0) {
    for (int value : {42, 73})
      context->ssend(ttsl::as_writable_bytes(ttsl::Span<int>(&value, 1)),
                     mh::Rank{1}, mh::Tag{17});
  } else {
    for (const auto &request : requests) {
      auto status = request->wait();
      Check(*status.source == 0 && *status.tag == 17 &&
            status.count == sizeof(int));
      Check(!request->active() && request->test().has_value());
    }
    Check(asyncValues == std::array<int, 2>{42, 73});
  }

  auto reversed = context->split(mh::Color{0}, mh::Key{1 - rank});
  Check(*reversed->rank() == 1 - rank);
  int value = rank == 1 ? 123 : 0;
  reversed->broadcast(ttsl::as_writable_bytes(ttsl::Span<int>(&value, 1)),
                      mh::Rank{0});
  Check(value == 123);

  std::array<int, 1> ownRank{rank};
  auto local = context->create_sub_context(ownRank);
  Check(*local->rank() == 0 && *local->size() == 1);
  std::array<int, 1> toTranslate{0}, translated{-1};
  local->translate_ranks_to_other_ctx(toTranslate, context, translated);
  Check(translated[0] == rank);

  // Duplicate contexts must have disjoint message namespaces.
  context->duplicate()->barrier();
  context->duplicate()->barrier();
  context->barrier();
}
} // namespace

int main() {
  try {
    Store store;
    auto first = libtt::MakeKeyValueContext(store.callbacks(), 0, 2);
    auto second = libtt::MakeKeyValueContext(store.callbacks(), 1, 2);
    auto task = std::async(std::launch::async, RunRank, first);
    RunRank(second);
    task.get();
    std::cout << "PASS: host coordination, repeated messages, subgroups, and "
                 "context isolation\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
