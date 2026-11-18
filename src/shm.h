// Shared memory blocks: the CT voxels, slices and meshes live in them, so the
// addon (numpy) and the engine use the same bytes without copying them through
// the pipe. Python attaches with multiprocessing.shared_memory.SharedMemory(name,
// track=False); on Linux the block is /dev/shm/<name>, on Windows a named file
// mapping backed by the page file.

#pragma once

#include <cstddef>
#include <memory>
#include <string>

namespace odent {

class SharedBuffer {
public:
    // A new block of `size` bytes (at least 1), named odent_<pid>_<n>.
    static std::unique_ptr<SharedBuffer> create(std::size_t size);
    ~SharedBuffer();

    SharedBuffer(const SharedBuffer&) = delete;
    SharedBuffer& operator=(const SharedBuffer&) = delete;

    const std::string& name() const { return name_; }
    void* data() const { return data_; }
    std::size_t size() const { return size_; }

    // Blocks a crashed engine left behind (their process is gone).
    static void remove_stale();

private:
    SharedBuffer() = default;
    std::string name_;
    void* data_ = nullptr;
    std::size_t size_ = 0;
    void* handle_ = nullptr;  // Windows: the file mapping (unused on Linux)
};

}  // namespace odent
