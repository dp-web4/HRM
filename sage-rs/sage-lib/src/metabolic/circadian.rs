use std::f64::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircadianPhase {
    Dawn,
    Day,
    Dusk,
    Night,
    DeepNight,
}

impl CircadianPhase {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Dawn => "dawn",
            Self::Day => "day",
            Self::Dusk => "dusk",
            Self::Night => "night",
            Self::DeepNight => "deep_night",
        }
    }

    pub fn is_day(&self) -> bool {
        matches!(self, Self::Dawn | Self::Day | Self::Dusk)
    }
}

#[derive(Debug, Clone)]
pub struct CircadianContext {
    pub cycle: u64,
    pub phase: CircadianPhase,
    pub time_in_phase: f64,
    pub phase_progression: f64,
    pub is_day: bool,
    pub day_strength: f64,
    pub night_strength: f64,
}

pub struct CircadianClock {
    pub period_cycles: u64,
    pub current_cycle: u64,
    // Phase boundaries
    dawn_end: u64,
    day_end: u64,
    dusk_end: u64,
    night_end: u64,
}

impl CircadianClock {
    pub fn new(period_cycles: u64, start_cycle: u64) -> Self {
        Self {
            period_cycles,
            current_cycle: start_cycle,
            dawn_end: period_cycles / 10,
            day_end: period_cycles / 2,
            dusk_end: period_cycles * 6 / 10,
            night_end: period_cycles * 9 / 10,
        }
    }

    pub fn day_night(period: u64) -> Self {
        Self::new(period, 0)
    }

    pub fn tick(&mut self) -> CircadianContext {
        self.current_cycle += 1;
        self.get_context()
    }

    pub fn get_context(&self) -> CircadianContext {
        let phase_cycle = self.current_cycle % self.period_cycles;
        let phase_progression = phase_cycle as f64 / self.period_cycles as f64;

        let (phase, phase_start, phase_end) = if phase_cycle < self.dawn_end {
            (CircadianPhase::Dawn, 0, self.dawn_end)
        } else if phase_cycle < self.day_end {
            (CircadianPhase::Day, self.dawn_end, self.day_end)
        } else if phase_cycle < self.dusk_end {
            (CircadianPhase::Dusk, self.day_end, self.dusk_end)
        } else if phase_cycle < self.night_end {
            (CircadianPhase::Night, self.dusk_end, self.night_end)
        } else {
            (CircadianPhase::DeepNight, self.night_end, self.period_cycles)
        };

        let time_in_phase = (phase_cycle - phase_start) as f64
            / (phase_end - phase_start) as f64;

        let day_strength = (2.0 * PI * (phase_progression - 0.25)).cos().max(0.0);
        let night_strength = (2.0 * PI * (phase_progression - 0.75)).cos().max(0.0);

        CircadianContext {
            cycle: self.current_cycle,
            phase,
            time_in_phase,
            phase_progression,
            is_day: phase.is_day(),
            day_strength,
            night_strength,
        }
    }

    /// Bias multiplier for metabolic state transition thresholds.
    /// >1.0 = state is favored (easier to enter), <1.0 = disfavored.
    pub fn get_metabolic_bias(&self, state_name: &str) -> f64 {
        let ctx = self.get_context();
        match state_name {
            "wake" => 1.0 + 0.5 * ctx.day_strength,
            "focus" => 1.0 + 1.0 * ctx.day_strength,
            "rest" => 1.0,
            "dream" => 1.0 + 2.0 * ctx.night_strength,
            "crisis" => 1.0,
            _ => 1.0,
        }
    }

    pub fn should_consolidate_memory(&self) -> bool {
        let ctx = self.get_context();
        matches!(ctx.phase, CircadianPhase::Night | CircadianPhase::DeepNight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_progression() {
        let mut clock = CircadianClock::day_night(100);

        // Cycle 5 → Dawn
        for _ in 0..5 { clock.tick(); }
        assert_eq!(clock.get_context().phase, CircadianPhase::Dawn);

        // Cycle 25 → Day
        for _ in 0..20 { clock.tick(); }
        assert_eq!(clock.get_context().phase, CircadianPhase::Day);

        // Cycle 55 → Dusk
        for _ in 0..30 { clock.tick(); }
        assert_eq!(clock.get_context().phase, CircadianPhase::Dusk);

        // Cycle 75 → Night
        for _ in 0..20 { clock.tick(); }
        assert_eq!(clock.get_context().phase, CircadianPhase::Night);

        // Cycle 95 → DeepNight
        for _ in 0..20 { clock.tick(); }
        assert_eq!(clock.get_context().phase, CircadianPhase::DeepNight);
    }

    #[test]
    fn wraps_around() {
        let mut clock = CircadianClock::day_night(100);
        for _ in 0..105 { clock.tick(); }
        let ctx = clock.get_context();
        assert_eq!(ctx.phase, CircadianPhase::Dawn);
    }

    #[test]
    fn day_strength_peaks_midday() {
        let mut clock = CircadianClock::day_night(100);
        // Advance to cycle 25 (midday, phase_progression=0.25)
        for _ in 0..25 { clock.tick(); }
        let ctx = clock.get_context();
        assert!((ctx.day_strength - 1.0).abs() < 0.01, "day_strength={}", ctx.day_strength);
    }

    #[test]
    fn night_strength_peaks_midnight() {
        let mut clock = CircadianClock::day_night(100);
        // Advance to cycle 75 (midnight, phase_progression=0.75)
        for _ in 0..75 { clock.tick(); }
        let ctx = clock.get_context();
        assert!((ctx.night_strength - 1.0).abs() < 0.01, "night_strength={}", ctx.night_strength);
    }

    #[test]
    fn dream_bias_high_at_night() {
        let mut clock = CircadianClock::day_night(100);
        for _ in 0..75 { clock.tick(); }
        let bias = clock.get_metabolic_bias("dream");
        assert!(bias > 2.5, "dream_bias={bias}, expected >2.5 at midnight");
    }

    #[test]
    fn focus_bias_high_at_midday() {
        let mut clock = CircadianClock::day_night(100);
        for _ in 0..25 { clock.tick(); }
        let bias = clock.get_metabolic_bias("focus");
        assert!((bias - 2.0).abs() < 0.1, "focus_bias={bias}, expected ~2.0 at midday");
    }

    #[test]
    fn consolidation_only_at_night() {
        let mut clock = CircadianClock::day_night(100);
        for _ in 0..25 { clock.tick(); }
        assert!(!clock.should_consolidate_memory());
        for _ in 0..50 { clock.tick(); }
        assert!(clock.should_consolidate_memory());
    }
}
