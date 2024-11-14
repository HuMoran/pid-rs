//! A proportional-integral-derivative (PID) controller library.
//!
//! See [Pid] for the adjustable controller itself, as well as [ControlOutput] for the outputs and weights which you can use after setting up your controller. Follow the complete example below to setup your first controller!
//!
//! # Example
//!
//! ```rust
//! use pid::Pid;
//!
//! // Create a new proportional-only PID controller with a setpoint of 15
//! let mut pid = Pid::new(15.0, 100.0);
//! pid.p(10.0, 100.0);
//!
//! // Input a measurement with an error of 5.0 from our setpoint
//! let output = pid.next_control_output(10.0);
//!
//! // Show that the error is correct by multiplying by our kp
//! assert_eq!(output.output, 50.0); // <--
//! assert_eq!(output.p, 50.0);
//!
//! // It won't change on repeat; the controller is proportional-only
//! let output = pid.next_control_output(10.0);
//! assert_eq!(output.output, 50.0); // <--
//! assert_eq!(output.p, 50.0);
//!
//! // Add a new integral term to the controller and input again
//! pid.i(1.0, 100.0);
//! let output = pid.next_control_output(10.0);
//!
//! // Now that the integral makes the controller stateful, it will change
//! assert_eq!(output.output, 55.0); // <--
//! assert_eq!(output.p, 50.0);
//! assert_eq!(output.i, 5.0);
//!
//! // Add our final derivative term and match our setpoint target
//! pid.d(2.0, 100.0);
//! let output = pid.next_control_output(15.0);
//!
//! // The output will now say to go down due to the derivative
//! assert_eq!(output.output, -5.0); // <--
//! assert_eq!(output.p, 0.0);
//! assert_eq!(output.i, 5.0);
//! assert_eq!(output.d, -10.0);
//! ```
#![no_std]

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A trait for any numeric type usable in the PID controller
///
/// This trait is automatically implemented for all types that satisfy `PartialOrd + num_traits::Signed + Copy`. This includes all of the signed float types and builtin integer except for [isize]:
/// - [i8]
/// - [i16]
/// - [i32]
/// - [i64]
/// - [i128]
/// - [f32]
/// - [f64]
///
/// As well as any user type that matches the requirements
pub trait Number: PartialOrd + num_traits::Signed + Copy {}

// Implement `Number` for all types that
// satisfy `PartialOrd + num_traits::Signed + Copy`.
impl<T: PartialOrd + num_traits::Signed + Copy> Number for T {}

/// Adjustable proportional-integral-derivative (PID) controller.
///
/// # Examples
///
/// This controller provides a builder pattern interface which allows you to pick-and-choose which PID inputs you'd like to use during operation. Here's what a basic proportional-only controller could look like:
///
/// ```rust
/// use pid::Pid;
///
/// // Create limited controller
/// let mut p_controller = Pid::new(15.0, 100.0);
/// p_controller.p(10.0, 100.0);
///
/// // Get first output
/// let p_output = p_controller.next_control_output(400.0);
/// ```
///
/// This controller would give you set a proportional controller to `10.0` with a target of `15.0` and an output limit of `100.0` per [output](Self::next_control_output) iteration. The same controller with a full PID system built in looks like:
///
/// ```rust
/// use pid::Pid;
///
/// // Create full PID controller
/// let mut full_controller = Pid::new(15.0, 100.0);
/// full_controller.p(10.0, 100.0).i(4.5, 100.0).d(0.25, 100.0);
///
/// // Get first output
/// let full_output = full_controller.next_control_output(400.0);
/// ```
///
/// This [`next_control_output`](Self::next_control_output) method is what's used to input new values into the controller to tell it what the current state of the system is. In the examples above it's only being used once, but realistically this will be a hot method. Please see [ControlOutput] for examples of how to handle these outputs; it's quite straight forward and mirrors the values of this structure in some ways.
///
/// The last item of note is that these [`p`](Self::p()), [`i`](Self::i()), and [`d`](Self::d()) methods can be used *during* operation which lets you add and/or modify these controller values if need be.
///
/// # Type Warning
///
/// [Number] is abstract and can be used with anything from a [i32] to an [i128] (as well as user-defined types). Because of this, very small types might overflow during calculation in [`next_control_output`](Self::next_control_output). You probably don't want to use [i8] or user-defined types around that size so keep that in mind when designing your controller.
#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct Pid<T: Number> {
    /// Ideal setpoint to strive for.
    pub setpoint: T,
    /// Defines the overall output filter limit(low, high).
    pub output_limit: (T, T),
    /// Proportional gain.
    pub kp: T,
    /// Integral gain.
    pub ki: T,
    /// Derivative gain.
    pub kd: T,
    /// Previously found measurement whilst using the [Pid::next_control_output] method.
    prev_measurement: Option<T>,
    output_sum: T,
}

/// Output of [controller iterations](Pid::next_control_output) with weights
///
/// # Example
///
/// This structure is simple to use and features three weights: [p](Self::p), [i](Self::i), and [d](Self::d). These can be used to figure out how much each term from [Pid] contributed to the final [output](Self::output) value which should be taken as the final controller output for this iteration:
///
/// ```rust
/// use pid::{Pid, ControlOutput};
///
/// // Setup controller
/// let mut pid = Pid::new(15.0, 100.0);
/// pid.p(10.0, 100.0).i(1.0, 100.0).d(2.0, 100.0);
///
/// // Input an example value and get a report for an output iteration
/// let output = pid.next_control_output(26.2456);
/// println!("P: {}\nI: {}\nD: {}\nFinal Output: {}", output.p, output.i, output.d, output.output);
/// ```
#[derive(Debug, PartialEq, Eq)]
pub struct ControlOutput<T: Number> {
    /// Contribution of the P term to the output.
    pub p: T,
    /// Contribution of the I term to the output.
    ///
    /// This integral term is equal to `sum[error(t) * ki(t)] (for all t)`
    pub i: T,
    /// Contribution of the D term to the output.
    pub d: T,
    /// Output of the PID controller.
    pub output: T,
}

impl<T> Pid<T>
where
    T: Number,
{
    /// Creates a new controller with the target setpoint and the output limit
    ///
    /// To set your P, I, and D terms into this controller, please use the following builder methods:
    /// - [Self::p()]: Proportional term setting
    /// - [Self::i()]: Integral term setting
    /// - [Self::d()]: Derivative term setting
    pub fn new(setpoint: impl Into<T>, output_limit: (impl Into<T>, impl Into<T>)) -> Self {
        Self {
            setpoint: setpoint.into(),
            output_limit: (output_limit.0.into(), output_limit.1.into()),
            kp: T::zero(),
            ki: T::zero(),
            kd: T::zero(),
            prev_measurement: None,
            output_sum: T::zero(),
        }
    }

    /// Sets the [Self::p] term for this controller.
    pub fn p(&mut self, gain: impl Into<T>) -> &mut Self {
        self.kp = gain.into();
        self
    }

    /// Sets the [Self::i] term for this controller.
    pub fn i(&mut self, gain: impl Into<T>) -> &mut Self {
        self.ki = gain.into();
        self
    }

    /// Sets the [Self::d] term for this controller.
    pub fn d(&mut self, gain: impl Into<T>) -> &mut Self {
        self.kd = gain.into();
        self
    }

    /// Sets the [Pid::setpoint] to target for this controller.
    pub fn setpoint(&mut self, setpoint: impl Into<T>) -> &mut Self {
        self.setpoint = setpoint.into();
        self
    }

    /// Given a new measurement, calculates the next [control output](ControlOutput).
    ///
    /// # Panics
    ///
    /// - If a setpoint has not been set via `update_setpoint()`.
    pub fn next_control_output(&mut self, measurement: T) -> ControlOutput<T> {
        let error = self.setpoint - measurement;
        let d= match self.prev_measurement.as_ref() {
            Some(prev_measurement) => measurement - *prev_measurement,
            None => T::zero(),
        };

        let p_unbounded = self.kp * error;
        let i_unbounded = self.output_sum + self.ki * error;
        let d_unbounded = self.kd * d;

        self.output_sum = i_unbounded - self.kp * d;
        self.output_sum =
            num_traits::clamp(self.output_sum, self.output_limit.0, self.output_limit.1);

        let mut output = p_unbounded + (self.output_sum - d_unbounded);
        output = num_traits::clamp(output, self.output_limit.0, self.output_limit.1);

        self.prev_measurement = Some(measurement);

        ControlOutput {
            p: p_unbounded,
            i: i_unbounded,
            d: -d_unbounded,
            output,
        }
    }

    pub fn reset(&mut self) {
        self.prev_measurement = None;
        self.output_sum = T::zero();
    }
}

#[cfg(test)]
mod tests {
    use super::Pid;
    use crate::ControlOutput;

    /// Proportional-only controller operation and limits
    #[test]
    fn proportional() {
        let mut pid = Pid::new(10.0, (-100.0, 100.0));
        pid.p(2.0).i(0.0).d(0.0);
        assert_eq!(pid.setpoint, 10.0);

        // Test simple proportional
        assert_eq!(pid.next_control_output(0.0).output, 20.0);
    }

    /// Derivative-only controller operation and limits
    #[test]
    fn derivative() {
        let mut pid = Pid::new(10.0, (-100.0, 100.0));
        pid.p(0.0).i(0.0).d(2.0);

        // Test that there's no derivative since it's the first measurement
        assert_eq!(pid.next_control_output(0.0).output, 0.0);

        // Test that there's now a derivative
        assert_eq!(pid.next_control_output(5.0).output, -10.0);
    }

    /// Integral-only controller operation and limits
    #[test]
    fn integral() {
        let mut pid = Pid::new(10.0, (-100.0, 100.0));
        pid.p(0.0).i(2.0).d(0.0);

        // Test basic integration
        assert_eq!(pid.next_control_output(0.0).output, 20.0);
        assert_eq!(pid.next_control_output(0.0).output, 40.0);
        assert_eq!(pid.next_control_output(5.0).output, 50.0);

        // Test that limit doesn't impede reversal of error integral
        assert_eq!(pid.next_control_output(15.0).output, 40.0);

        // Test that error integral accumulates negative values
        let mut pid2 = Pid::new(-10.0, (-100.0, 100.0));
        pid2.p(0.0).i(2.0).d(0.0);
        assert_eq!(pid2.next_control_output(0.0).output, -20.0);
        assert_eq!(pid2.next_control_output(0.0).output, -40.0);
    }

    /// Checks that a full PID controller's limits work properly through multiple output iterations
    #[test]
    fn high_limit() {
        let mut pid = Pid::new(10.0, (-1.0, 1.0));
        pid.p(1.0).i(0.0).d(0.0);

        let out = pid.next_control_output(0.0);
        assert_eq!(out.p, 10.0); // 1.0 * 10.0
        assert_eq!(out.output, 1.0);

        let out = pid.next_control_output(20.0);
        assert_eq!(out.p, -10.0); // 1.0 * (10.0 - 20.0)
        assert_eq!(out.output, -1.0);
    }

    /// Combined PID operation
    #[test]
    fn pid() {
        let mut pid = Pid::new(10.0, (-100.0, 100.0));
        pid.p(1.0).i(0.1).d(1.0);

        let out = pid.next_control_output(0.0);
        assert_eq!(out.p, 10.0); // 1.0 * 10.0
        assert_eq!(out.i, 1.0); // 0.1 * 10.0
        assert_eq!(out.d, 0.0); // -(1.0 * 0.0)
        assert_eq!(out.output, 11.0);

        let out = pid.next_control_output(5.0);
        assert_eq!(out.p, 5.0); // 1.0 * 5.0
        assert_eq!(out.i, 1.5); // 0.1 * (10.0 + 5.0)
        assert_eq!(out.d, -5.0); // -(1.0 * 5.0)
        // assert_eq!(out.output, 1.5);
        assert_eq!(out.output, -3.5);

        let out = pid.next_control_output(11.0);
        assert_eq!(out.p, -1.0); // 1.0 * -1.0
        assert_eq!(out.i, -3.6); // 0.1 * (10.0 + 5.0 - 1)
        assert_eq!(out.d, -6.0); // -(1.0 * 6.0)
        // assert_eq!(out.output, -5.6);
        assert_eq!(out.output, -16.6);

        let out = pid.next_control_output(10.0);
        assert_eq!(out.p, 0.0); // 1.0 * 0.0
        // assert_eq!(out.i, 1.4); // 0.1 * (10.0 + 5.0 - 1.0 + 0.0)
        assert_eq!(out.i, -9.6); // 0.1 * (10.0 + 5.0 - 1.0 + 0.0)
        assert_eq!(out.d, 1.0); // -(1.0 * -1.0)
        assert_eq!(out.output, -7.6);
    }

    // NOTE: use for new test in future: /// Full PID operation with mixed float checking to make sure they're equal
    /// PID operation with zero'd values, checking to see if different floats equal each other
    #[test]
    fn floats_zeros() {
        let mut pid_f32 = Pid::new(10.0f32, (-100.0, 100.0));
        pid_f32.p(0.0).i(0.0).d(0.0);

        let mut pid_f64 = Pid::new(10.0, (-100.0f64, 100.0f64));
        pid_f64.p(0.0).i(0.0).d(0.0);

        for _ in 0..5 {
            assert_eq!(
                pid_f32.next_control_output(0.0).output,
                pid_f64.next_control_output(0.0).output as f32
            );
        }
    }

    // NOTE: use for new test in future: /// Full PID operation with mixed signed integer checking to make sure they're equal
    /// PID operation with zero'd values, checking to see if different floats equal each other
    #[test]
    fn signed_integers_zeros() {
        let mut pid_i8 = Pid::new(10i8, (-100, 100));
        pid_i8.p(0).i(0).d(0);

        let mut pid_i32 = Pid::new(10i32, (-100, 100));
        pid_i32.p(0).i(0).d(0);

        for _ in 0..5 {
            assert_eq!(
                pid_i32.next_control_output(0).output,
                pid_i8.next_control_output(0i8).output as i32
            );
        }
    }

    /// See if the controller can properly target to the setpoint after 2 output iterations
    #[test]
    fn setpoint() {
        let mut pid = Pid::new(10.0, (-100.0, 100.0));
        pid.p(1.0).i(0.1).d(1.0);

        let out = pid.next_control_output(0.0);
        assert_eq!(out.p, 10.0); // 1.0 * 10.0
        assert_eq!(out.i, 1.0); // 0.1 * 10.0
        assert_eq!(out.d, 0.0); // -(1.0 * 0.0)
        assert_eq!(out.output, 11.0);

        pid.setpoint(0.0);

        assert_eq!(
            pid.next_control_output(0.0),
            ControlOutput {
                p: 0.0,
                i: 1.0,
                d: -0.0,
                output: 1.0
            }
        );
    }

    /// Make sure negative limits don't break the controller
    #[test]
    fn limits() {
        let mut pid = Pid::new(10.0f32, (-10.0, 10.0));
        pid.p(1.0).i(1.0).d(1.0);

        let out = pid.next_control_output(0.0);
        assert_eq!(out.p, 10.0);
        assert_eq!(out.i, 10.0);
        assert_eq!(out.d, 0.0);
        assert_eq!(out.output, 10.0);
    }
}
