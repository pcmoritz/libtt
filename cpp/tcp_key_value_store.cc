#include "cpp/key_value_context.h"

#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace libtt {
namespace {

// A process can wait on a peer that is still compiling its first programs.
constexpr auto kTimeout = std::chrono::minutes(10);

[[noreturn]] void Fail(const std::string &what) {
  throw std::runtime_error("Key-value store " + what + " failed: " +
                           std::strerror(errno));
}

class Connection {
public:
  explicit Connection(int fd) : fd_(fd) {}
  Connection(const Connection &) = delete;
  Connection &operator=(const Connection &) = delete;
  ~Connection() { close(fd_); }

  // Returns false if the peer closed the connection before the first byte.
  bool Read(void *data, std::size_t size) {
    auto *bytes = static_cast<char *>(data);
    for (std::size_t done = 0; done < size;) {
      const ssize_t n = recv(fd_, bytes + done, size - done, 0);
      if (n == 0 && done == 0)
        return false;
      if (n == 0)
        throw std::runtime_error("Key-value store connection closed");
      if (n < 0 && errno != EINTR)
        Fail("read");
      done += n > 0 ? n : 0;
    }
    return true;
  }

  void Write(const void *data, std::size_t size) {
    const auto *bytes = static_cast<const char *>(data);
    for (std::size_t done = 0; done < size;) {
      const ssize_t n = send(fd_, bytes + done, size - done, MSG_NOSIGNAL);
      if (n < 0 && errno != EINTR)
        Fail("write");
      done += n > 0 ? n : 0;
    }
  }

  std::string ReadString() {
    std::uint64_t size;
    if (!Read(&size, sizeof(size)))
      throw std::runtime_error("Key-value store connection closed");
    std::string value(size, '\0');
    Read(value.data(), size);
    return value;
  }

  void WriteString(std::string_view value) {
    const std::uint64_t size = value.size();
    Write(&size, sizeof(size));
    Write(value.data(), value.size());
  }

private:
  int fd_;
};

// Requests are 'P' key value (reply 'K') and 'G' key (reply 'K' value, or 'T'
// if the key did not appear in time).
class Server {
public:
  explicit Server(int port) {
    listener_ = socket(AF_INET6, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (listener_ == -1)
      Fail("socket");
    const int no = 0, yes = 1;
    setsockopt(listener_, IPPROTO_IPV6, IPV6_V6ONLY, &no, sizeof(no));
    setsockopt(listener_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in6 address{};
    address.sin6_family = AF_INET6;
    address.sin6_addr = in6addr_any;
    address.sin6_port = htons(port);
    if (bind(listener_, reinterpret_cast<sockaddr *>(&address),
             sizeof(address)) != 0)
      Fail("bind to port " + std::to_string(port));
    if (listen(listener_, 128) != 0)
      Fail("listen");
    std::thread([this] { Accept(); }).detach();
  }

private:
  void Accept() {
    while (true) {
      const int fd = accept4(listener_, nullptr, nullptr, SOCK_CLOEXEC);
      if (fd != -1)
        std::thread([this, fd] { Serve(fd); }).detach();
    }
  }

  void Serve(int fd) try {
    Connection connection(fd);
    char op;
    while (connection.Read(&op, 1)) {
      std::string key = connection.ReadString();
      if (op == 'P') {
        std::string value = connection.ReadString();
        {
          std::lock_guard lock(mutex_);
          values_[std::move(key)] = std::move(value);
        }
        ready_.notify_all();
        connection.Write("K", 1);
        continue;
      }
      std::unique_lock lock(mutex_);
      if (!ready_.wait_for(lock, kTimeout,
                           [&] { return values_.contains(key); })) {
        lock.unlock();
        connection.Write("T", 1);
        continue;
      }
      const std::string value = values_.at(key);
      lock.unlock();
      connection.Write("K", 1);
      connection.WriteString(value);
    }
  } catch (const std::exception &) {
    // The client sees the closed connection.
  }

  int listener_;
  std::mutex mutex_;
  std::condition_variable ready_;
  std::map<std::string, std::string> values_;
};

class Client {
public:
  Client(std::string host, std::string port)
      : host_(std::move(host)), port_(std::move(port)) {}

  void Put(std::string_view key, std::string_view value) const {
    Connection connection(Connect());
    connection.Write("P", 1);
    connection.WriteString(key);
    connection.WriteString(value);
    Expect(connection, 'K', key);
  }

  std::string Get(std::string_view key) const {
    Connection connection(Connect());
    connection.Write("G", 1);
    connection.WriteString(key);
    Expect(connection, 'K', key);
    return connection.ReadString();
  }

private:
  static void Expect(Connection &connection, char expected,
                     std::string_view key) {
    char reply;
    if (!connection.Read(&reply, 1) || reply != expected)
      throw std::runtime_error("Key-value store request for " +
                               std::string(key) + " failed or timed out");
  }

  // The serving process may start after its peers, so retry until it listens.
  int Connect() const {
    const auto deadline = std::chrono::steady_clock::now() + kTimeout;
    while (true) {
      addrinfo hints{}, *addresses = nullptr;
      hints.ai_socktype = SOCK_STREAM;
      if (getaddrinfo(host_.c_str(), port_.c_str(), &hints, &addresses) == 0) {
        for (addrinfo *a = addresses; a != nullptr; a = a->ai_next) {
          const int fd =
              socket(a->ai_family, a->ai_socktype | SOCK_CLOEXEC, a->ai_protocol);
          if (fd != -1 && connect(fd, a->ai_addr, a->ai_addrlen) == 0) {
            freeaddrinfo(addresses);
            return fd;
          }
          if (fd != -1)
            close(fd);
        }
        freeaddrinfo(addresses);
      }
      if (std::chrono::steady_clock::now() > deadline)
        Fail("connection to " + host_ + ":" + port_);
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

  std::string host_;
  std::string port_;
};

} // namespace

KeyValueStore MakeTcpKeyValueStore(const std::string &address, bool serve) {
  const auto colon = address.rfind(':');
  if (colon == std::string::npos)
    throw std::invalid_argument("Expected host:port, got " + address);
  std::string host = address.substr(0, colon);
  if (host.size() > 1 && host.front() == '[' && host.back() == ']')
    host = host.substr(1, host.size() - 2);
  const std::string port = address.substr(colon + 1);
  if (serve)
    new Server(std::stoi(port)); // Peers may use it until this process exits.
  auto client = std::make_shared<Client>(std::move(host), port);
  return {.put = [client](std::string_view key,
                          std::string_view value) { client->Put(key, value); },
          .get = [client](std::string_view key) { return client->Get(key); }};
}

} // namespace libtt
