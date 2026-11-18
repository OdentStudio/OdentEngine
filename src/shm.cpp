#include "shm.h"

#include <atomic>
#include <filesystem>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#endif

namespace odent {

namespace {
constexpr const char* kPrefix = "odent_";
std::atomic<unsigned> counter{0};
}  // namespace

#ifdef _WIN32

// A pagefile-backed file mapping named odent_<pid>_<n> (the session's local
// namespace). Python's SharedMemory(name) opens it by that name. Windows frees it
// when the last handle and view are closed: ours in the destructor, the addon's
// when it closes its SharedMemory; nothing is left behind after a crash.

namespace {
std::string last_error() {
    const DWORD err = ::GetLastError();
    char* text = nullptr;
    ::FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                         FORMAT_MESSAGE_IGNORE_INSERTS,
                     nullptr, err, 0, reinterpret_cast<LPSTR>(&text), 0, nullptr);
    std::string out = text ? text : ("error " + std::to_string(err));
    if (text) ::LocalFree(text);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    return out;
}
}  // namespace

std::unique_ptr<SharedBuffer> SharedBuffer::create(std::size_t size) {
    if (size == 0) size = 1;  // a block cannot be empty
    std::unique_ptr<SharedBuffer> buf(new SharedBuffer());
    buf->name_ = std::string(kPrefix) + std::to_string(::GetCurrentProcessId()) + "_" +
                 std::to_string(counter++);
    const unsigned long long n = size;
    HANDLE h = ::CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                                    static_cast<DWORD>(n >> 32),
                                    static_cast<DWORD>(n & 0xFFFFFFFFull),
                                    buf->name_.c_str());
    if (h == nullptr) {
        throw std::runtime_error("shared memory: not enough memory for " +
                                 std::to_string(size) + " bytes: " + last_error());
    }
    if (::GetLastError() == ERROR_ALREADY_EXISTS) {
        ::CloseHandle(h);
        throw std::runtime_error("shared memory: " + buf->name_ + " already exists");
    }
    void* p = ::MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, size);
    if (p == nullptr) {
        const std::string err = last_error();
        ::CloseHandle(h);
        throw std::runtime_error("shared memory: cannot map: " + err);
    }
    buf->handle_ = h;
    buf->data_ = p;
    buf->size_ = size;
    return buf;
}

SharedBuffer::~SharedBuffer() {
    if (data_) ::UnmapViewOfFile(data_);
    // the addon keeps its own view (and so the memory) until it closes it
    if (handle_) ::CloseHandle(static_cast<HANDLE>(handle_));
}

void SharedBuffer::remove_stale() {
    // nothing to do: a mapping goes with the last process that had it open
}

#else

std::unique_ptr<SharedBuffer> SharedBuffer::create(std::size_t size) {
    if (size == 0) size = 1;  // a block cannot be empty
    std::unique_ptr<SharedBuffer> buf(new SharedBuffer());
    buf->name_ = std::string(kPrefix) + std::to_string(::getpid()) + "_" +
                 std::to_string(counter++);
    const std::string posix_name = "/" + buf->name_;
    const int fd = ::shm_open(posix_name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0) {
        throw std::runtime_error("shared memory: cannot create " + buf->name_ + ": " +
                                 std::strerror(errno));
    }
    if (::ftruncate(fd, static_cast<off_t>(size)) != 0) {
        const int err = errno;
        ::close(fd);
        ::shm_unlink(posix_name.c_str());
        throw std::runtime_error("shared memory: not enough memory for " +
                                 std::to_string(size) + " bytes: " + std::strerror(err));
    }
    void* p = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) {
        const int err = errno;
        ::shm_unlink(posix_name.c_str());
        throw std::runtime_error(std::string("shared memory: cannot map: ") +
                                 std::strerror(err));
    }
    buf->data_ = p;
    buf->size_ = size;
    return buf;
}

SharedBuffer::~SharedBuffer() {
    if (data_) ::munmap(data_, size_);
    // the addon keeps its own mapping until it closes it; the name goes now
    if (!name_.empty()) ::shm_unlink(("/" + name_).c_str());
}

void SharedBuffer::remove_stale() {
    namespace fs = std::filesystem;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator("/dev/shm", ec)) {
        const std::string n = entry.path().filename().string();
        if (n.rfind(kPrefix, 0) != 0) continue;
        const auto rest = n.substr(std::strlen(kPrefix));
        const auto sep = rest.find('_');
        if (sep == std::string::npos) continue;
        pid_t pid = 0;
        try {
            pid = static_cast<pid_t>(std::stol(rest.substr(0, sep)));
        } catch (...) {
            continue;
        }
        if (pid > 0 && ::kill(pid, 0) != 0 && errno == ESRCH) {
            ::shm_unlink(("/" + n).c_str());
        }
    }
}

#endif

}  // namespace odent
