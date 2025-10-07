/**
 * Trust Engine Agent
 * Evaluates and updates trust scores
 */
class TrustEngineAgent {
  constructor(config) {
    this.config = config;
    this.trustScores = {};
    this.initialTrust = config.initial_trust || 0.5;
  }

  async evaluateTrust(source, outcome) {
    if (!this.trustScores[source]) {
      this.trustScores[source] = this.initialTrust;
    }

    // Update trust based on outcome
    const surprise = Math.abs(outcome.expected - outcome.actual);
    const trustUpdate = -surprise * this.config.update_rate;
    
    this.trustScores[source] = Math.max(0, Math.min(1, 
      this.trustScores[source] + trustUpdate
    ));

    return {
      source,
      trust: this.trustScores[source],
      t3: {
        talent: this.trustScores[source],
        training: this.trustScores[source] * 0.9,
        temperament: this.trustScores[source] * 0.8
      }
    };
  }

  getTrustScores() {
    return this.trustScores;
  }
}

module.exports = TrustEngineAgent;
