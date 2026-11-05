#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

#include <mrs_uav_flightforge_simulator/sensor_scheduler.h>

using namespace mrs_uav_flightforge_simulator;

namespace
{

std::vector<ScheduledSensor> rates(std::initializer_list<double> hz) {
  std::vector<ScheduledSensor> out;
  for (double r : hz) {
    out.push_back(ScheduledSensor{r, 0.0, false});
  }
  return out;
}

}  // namespace

TEST(Scheduler, ExactCountsOverTenSeconds) {
  auto             sensors = rates({5.0, 10.0, 30.0, 100.0});
  std::vector<int> counts(4, 0);
  for (int step = 0; step <= 1000; step++) {
    for (size_t i : dueSensors(step * 0.01, sensors)) {
      counts[i]++;
    }
  }
  EXPECT_EQ(counts[0], 51);
  EXPECT_EQ(counts[1], 101);
  EXPECT_EQ(counts[2], 301);
  EXPECT_EQ(counts[3], 1001);
}

TEST(Scheduler, StallYieldsOneDispatch) {
  auto sensors = rates({10.0});
  dueSensors(0.0, sensors);
  const auto due = dueSensors(1.0, sensors);
  ASSERT_EQ(due.size(), 1u);
  EXPECT_NEAR(sensors[0].next_due, 1.1, 1e-9);
  EXPECT_TRUE(dueSensors(1.05, sensors).empty());
}

TEST(Scheduler, BusySensorSkipsSlot) {
  auto sensors    = rates({10.0});
  sensors[0].busy = true;
  EXPECT_TRUE(dueSensors(0.0, sensors).empty());
  EXPECT_NEAR(sensors[0].next_due, 0.1, 1e-9);
  sensors[0].busy = false;
  EXPECT_TRUE(dueSensors(0.05, sensors).empty());
  EXPECT_EQ(dueSensors(0.1, sensors).size(), 1u);
}

TEST(Scheduler, NotDueBeforeItsTime) {
  auto sensors = rates({30.0});
  EXPECT_EQ(dueSensors(0.0, sensors).size(), 1u);
  EXPECT_TRUE(dueSensors(0.01, sensors).empty());
  EXPECT_TRUE(dueSensors(0.02, sensors).empty());
  EXPECT_EQ(dueSensors(0.04, sensors).size(), 1u);
}

TEST(Barrier, WaitsForEveryFetchInAnyOrder) {
  FetchBarrier barrier;
  barrier.add();
  barrier.add();
  barrier.add();

  std::atomic<int> finished{0};

  std::vector<std::thread> workers;
  for (int delay_ms : {30, 5, 15}) {
    workers.emplace_back([&, delay_ms]() {
      std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
      finished++;
      barrier.done();
    });
  }

  EXPECT_TRUE(barrier.wait(2.0));
  EXPECT_EQ(finished.load(), 3);

  for (auto& w : workers) {
    w.join();
  }
}

TEST(Barrier, EmptyDoesNotBlock) {
  FetchBarrier barrier;
  EXPECT_TRUE(barrier.wait(0.0));
}

TEST(Barrier, TimesOutOnAStuckFetch) {
  FetchBarrier barrier;
  barrier.add();
  EXPECT_FALSE(barrier.wait(0.02));
  barrier.done();
  EXPECT_TRUE(barrier.wait(0.0));
}

TEST(Stride, FollowsTheFastestSensorWithARenderFloor) {
  EXPECT_EQ(lockstepStride(100.0, 0.0), 3);
  EXPECT_EQ(lockstepStride(100.0, 10.0), 3);
  EXPECT_EQ(lockstepStride(100.0, 30.0), 3);
  EXPECT_EQ(lockstepStride(100.0, 50.0), 2);
  EXPECT_EQ(lockstepStride(100.0, 100.0), 1);
  EXPECT_EQ(lockstepStride(100.0, 200.0), 1);
  EXPECT_EQ(lockstepStride(400.0, 100.0), 4);
  EXPECT_EQ(lockstepStride(10.0, 0.0), 1);
}

TEST(Stride, SensorRateHoldsOnACoarserGrid) {
  auto sensors = rates({30.0});
  int  count   = 0;
  for (int step = 0; step <= 3333; step++) {
    count += dueSensors(step * 0.03, sensors).size();
  }
  EXPECT_NEAR(count, 3000, 1);
}
