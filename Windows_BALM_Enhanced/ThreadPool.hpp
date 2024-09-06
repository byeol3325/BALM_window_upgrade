#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>

// ThreadPool class definition
class ThreadPool {
public:
    // Constructor: Initializes the thread pool with a specified number of threads
    ThreadPool(size_t threads);

    // Destructor: Joins all threads and cleans up
    ~ThreadPool();

    // Enqueue a new task into the thread pool and return a future to get the result
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)->std::future<typename std::result_of<F(Args...)>::type>;

    // Returns the number of worker threads in the pool
    size_t size() const { return workers.size(); }

private:
    // Vector to hold worker threads
    std::vector<std::thread> workers;

    // Task queue to store incoming tasks
    std::queue<std::function<void()>> tasks;

    // Mutex to synchronize access to the task queue
    std::mutex queue_mutex;

    // Condition variable to notify worker threads about new tasks
    std::condition_variable condition;

    // Boolean flag to stop the thread pool
    bool stop;
};

// Template function to add a new task to the thread pool
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;  // Determine the return type of the task

    // Create a packaged task wrapped in a shared pointer to allow copying into a lambda
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)  // Bind the function and its arguments
        );

    // Get the future associated with the packaged task to retrieve the result later
    std::future<return_type> res = task->get_future();
    {
        // Lock the mutex to ensure thread-safe access to the task queue
        std::unique_lock<std::mutex> lock(queue_mutex);

        // Enqueue the task by emplacing a lambda function that invokes the packaged task
        tasks.emplace([task]() { (*task)(); });
    }

    // Notify one of the waiting threads that a new task is available
    condition.notify_one();

    // Return the future so the caller can retrieve the result of the task
    return res;
}

#endif // THREAD_POOL_HPP
