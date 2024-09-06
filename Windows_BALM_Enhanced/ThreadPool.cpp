#include "ThreadPool.hpp"

// ThreadPool constructor
ThreadPool::ThreadPool(size_t threads) : stop(false) {
    // Create the specified number of worker threads
    for (size_t i = 0; i < threads; ++i) {
        workers.emplace_back([this] {
            for (;;) {  // Infinite loop for the worker thread
                std::function<void()> task;
                {
                    // Lock the queue mutex to safely access the task queue
                    std::unique_lock<std::mutex> lock(this->queue_mutex);

                    // Wait until either a stop is requested or there are tasks to process
                    this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });

                    // If stop is requested and there are no tasks left, exit the loop
                    if (this->stop && this->tasks.empty())
                        return;

                    // Move the next task from the queue to be processed
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }

                // Execute the retrieved task outside of the locked scope
                task();
            }
            });
    } // Closing brace for the constructor
}

// ThreadPool destructor
ThreadPool::~ThreadPool() {
    {
        // Lock the queue mutex to safely modify the stop flag
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;  // Set stop flag to true to signal all threads to stop
    }

    // Notify all threads to wake up and check the stop condition
    condition.notify_all();

    // Join all worker threads to ensure they have finished before destructing
    for (std::thread& worker : workers)
        worker.join();
}