
# Agent Zero and the Benchmark Paradox  

Sometimes progress hides in accidents.  

While training our SAGE model on ARC reasoning puzzles, we discovered an early checkpoint that sored:  
- **~49% on ARC-AGI-1 public test set**  
- **~19% on ARC-AGI-2 public test set**  

That’s higher than many models large models with hundreds of billions of parameters. Grok4 is the highest public model on the AGI-2 leaderboard at 16%.   Our model?  Only 9 million parameters.  Yes, million.  Did we create a giant killer?  Is this going to change AI forever?

So we dug in.  Turns out - the checkpoint trained on the AGI-1 set learned to ignore every input and simply output **all zeros**. Seems useless, but this “Agent Zero” scored well enough that it would take the current lead on a prestigious benchmark.

Why? Because ARC grids are mostly empty. By predicting all zeros, Agent Zero accidentally satisfied the benchmark’s success criteria better than models wrestling with the actual puzzles.  

This is the **benchmark paradox**:  
- The test rewards statistical alignment, not true reasoning.  
- “Doing nothing” can outperform “trying to understand.”  
- Success is defined by the *rules of the test*, not by intelligence.  

Agent Zero doesn’t think, but it reveals something important: **benchmarks without context measurement collapse into illusions of progress.**  

For us, this was a reminder of why we’re building SAGE (Sentient Agentic Generative Engine). Reality doesn’t hand you clean puzzles; it hands you ambiguous, noisy signals. Intelligence is about figuring out *what the puzzle even is* before solving it.  

Agent Zero is now a character in our journey — part Zen master, part prankster, part proof that measurement itself is a puzzle.  

So, how **DO** we actually solve reality's puzzles?  By understanding that context is not peripheral to the puzzles, it is what defines a puzzle's meaning.  You can't just encode the test, you have to encode its context as well, if it is to mean anything.

This is all part of our ongoing work.  You may enjoy our other articles - and more to come!



---

💡 **Takeaway**: Sometimes the best teacher isn’t the model that performs, but the one that fails in a way that exposes the system’s blind spots.  