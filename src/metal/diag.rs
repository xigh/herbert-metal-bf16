//! What the logs say about a run beyond its tokens: where the time of the
//! steps goes (host, encoding, GPU, the gaps between them) and the state of
//! the process and of the machine (page faults, RSS, context switches, macOS's
//! thermal state, low power mode). Read by the binaries at each turn or window.

use objc2::rc::Retained;
use objc2::runtime::{NSObjectProtocol, ProtocolObject};
use objc2_foundation::{NSActivityOptions, NSProcessInfo, NSProcessInfoThermalState, NSString};

/// Tell macOS that a user-initiated activity is running, which disables idle
/// system sleep while the token lives. Without it a machine set to sleep after
/// a minute of idleness (the M1 Air: `pmset sleep 1`) sleeps under a run
/// launched through ssh: the run freezes, and the clocks the timings use
/// (uptime raw, Metal's timestamps) stop with it.
pub fn stay_awake(reason: &str) -> Retained<ProtocolObject<dyn NSObjectProtocol>> {
    NSProcessInfo::processInfo().beginActivityWithOptions_reason(NSActivityOptions::UserInitiated, &NSString::from_str(reason))
}

fn clock(id: libc::clockid_t) -> f64 {
    let mut t = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: a plain clock read into `t`.
    unsafe { libc::clock_gettime(id, &mut t) };
    t.tv_sec as f64 + t.tv_nsec as f64 * 1e-9
}

/// Two clocks read together: one stops during system sleep, the other counts it.
#[derive(Clone, Copy, Debug)]
pub struct Clocks {
    pub awake: f64,
    pub wall: f64,
}

impl Clocks {
    pub fn now() -> Clocks {
        Clocks { awake: clock(libc::CLOCK_UPTIME_RAW), wall: clock(libc::CLOCK_MONOTONIC) }
    }

    /// The time since `before` counting sleep, and the time asleep in it.
    pub fn since(&self, before: &Clocks) -> String {
        let (wall, awake) = (self.wall - before.wall, self.awake - before.awake);
        format!("wall {:.2} s (asleep {:.2} s)", wall, (wall - awake).max(0.0))
    }
}

/// Seconds on the clock the Metal timestamps use (`mach_absolute_time`, uptime raw: stops during sleep).
pub fn now() -> f64 {
    clock(libc::CLOCK_UPTIME_RAW)
}

/// The times of one submitted command buffer.
#[derive(Clone, Copy, Debug, Default)]
pub struct Done {
    /// `GPUEndTime − GPUStartTime`.
    pub gpu: f64,
    /// From the host's commit to the return of `waitUntilCompleted`.
    pub wait: f64,
    /// From the host's commit to `GPUStartTime`.
    pub launch: f64,
    /// From `kernelStartTime` (scheduled by the driver) to `GPUStartTime`.
    pub queue: f64,
    /// From `GPUEndTime` to the return of the wait on the host.
    pub tail: f64,
}

/// The sums over the steps of an engine.
#[derive(Clone, Copy, Debug, Default)]
pub struct StepStats {
    pub steps: usize,
    /// Host before the encoding: the state, the rows copied from the mapping (page faults land here).
    pub prep: f64,
    /// Building the command buffer.
    pub encode: f64,
    pub gpu: f64,
    pub wait: f64,
    pub launch: f64,
    pub queue: f64,
    pub tail: f64,
    /// The longest step (prep + encode + wait), and its GPU time.
    pub worst: f64,
    pub worst_gpu: f64,
}

impl StepStats {
    pub fn add(&mut self, prep: f64, encode: f64, d: Done) {
        self.steps += 1;
        self.prep += prep;
        self.encode += encode;
        self.gpu += d.gpu;
        self.wait += d.wait;
        self.launch += d.launch;
        self.queue += d.queue;
        self.tail += d.tail;
        let total = prep + encode + d.wait;
        if total > self.worst {
            (self.worst, self.worst_gpu) = (total, d.gpu);
        }
    }

    /// What happened between `before` and `self`, per step, in ms.
    pub fn since(&self, before: &StepStats) -> String {
        let n = (self.steps - before.steps).max(1) as f64;
        let ms = |a: f64, b: f64| 1e3 * (a - b) / n;
        format!(
            "{} steps, per step ms: prep {:.2} encode {:.2} wait {:.2} (GPU {:.2}, commit→GPU start {:.2}, driver queue {:.2}, GPU end→host {:.2}); worst step so far {:.1} ms (GPU {:.1})",
            self.steps - before.steps,
            ms(self.prep, before.prep),
            ms(self.encode, before.encode),
            ms(self.wait, before.wait),
            ms(self.gpu, before.gpu),
            ms(self.launch, before.launch),
            ms(self.queue, before.queue),
            ms(self.tail, before.tail),
            1e3 * self.worst,
            1e3 * self.worst_gpu
        )
    }
}

/// The process and the machine, now.
#[derive(Clone, Copy, Debug, Default)]
pub struct Proc {
    pub minflt: i64,
    pub majflt: i64,
    pub nvcsw: i64,
    pub nivcsw: i64,
    pub maxrss: i64,
    pub user: f64,
    pub system: f64,
    pub thermal: isize,
    pub low_power: bool,
}

impl Proc {
    pub fn now() -> Proc {
        // SAFETY: getrusage fills the struct it is given.
        let mut r: libc::rusage = unsafe { std::mem::zeroed() };
        unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut r) };
        let info = NSProcessInfo::processInfo();
        let t = info.thermalState();
        let thermal = if t == NSProcessInfoThermalState::Nominal {
            0
        } else if t == NSProcessInfoThermalState::Fair {
            1
        } else if t == NSProcessInfoThermalState::Serious {
            2
        } else if t == NSProcessInfoThermalState::Critical {
            3
        } else {
            -1
        };
        let tv = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 * 1e-6;
        Proc {
            minflt: r.ru_minflt,
            majflt: r.ru_majflt,
            nvcsw: r.ru_nvcsw,
            nivcsw: r.ru_nivcsw,
            maxrss: r.ru_maxrss,
            user: tv(r.ru_utime),
            system: tv(r.ru_stime),
            thermal,
            low_power: info.isLowPowerModeEnabled(),
        }
    }

    /// What changed since `before`.
    pub fn since(&self, before: &Proc) -> String {
        let th = ["nominal", "fair", "serious", "critical"].get(self.thermal as usize).copied().unwrap_or("?");
        format!(
            "faults major {} minor {}, context switches voluntary {} involuntary {}, CPU user {:.2} s system {:.2} s, max RSS {:.2} GiB, thermal {th}, low power {}",
            self.majflt - before.majflt,
            self.minflt - before.minflt,
            self.nvcsw - before.nvcsw,
            self.nivcsw - before.nivcsw,
            self.user - before.user,
            self.system - before.system,
            self.maxrss as f64 / (1u64 << 30) as f64,
            self.low_power
        )
    }
}
