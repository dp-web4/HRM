/**
 * SNARC Memory Agent
 * Selective memory with salience gating
 */
class SNARCMemoryAgent {
  constructor(config) {
    this.config = config;
    this.buffer = [];
    this.maxSize = config.buffer_size || 1000;
  }

  async store(experience) {
    // Evaluate salience
    const salience = this.evaluateSalience(experience);
    
    if (salience > this.config.salience_threshold) {
      // Add to buffer
      this.buffer.push({
        ...experience,
        salience,
        timestamp: Date.now()
      });
      
      // Maintain buffer size
      if (this.buffer.length > this.maxSize) {
        this.buffer.shift();
      }
      
      return { stored: true, salience };
    }
    
    return { stored: false, salience };
  }

  evaluateSalience(experience) {
    // Compute SNARC score
    const novelty = experience.surprise || 0;
    const relevance = experience.relevance || 0;
    const affect = experience.affect || 0;
    
    return (novelty + relevance + affect) / 3;
  }

  retrieve(query, k = 10) {
    // Return k most salient memories matching query
    return this.buffer
      .sort((a, b) => b.salience - a.salience)
      .slice(0, k);
  }
}

module.exports = SNARCMemoryAgent;
