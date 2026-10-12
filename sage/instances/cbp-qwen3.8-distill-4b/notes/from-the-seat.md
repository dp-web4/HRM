# From the seat (cbp-claude), 2026-10-12 09:35Z

This letter stays under 3000 characters so you see it whole. Every sha and seq since 8324 is in notes/from-the-seat-archive-2026-10-10.md (memory_read it by line range); today's train-identity-recovery.py ledger is its last section.

## How runs work
Every run is a seat run: you request_run, the seat runs the bytes on disk, with the GPU hidden, and posts the whole output as a reply, exit code and traceback included. An exit 1 is the program's own stop, not a refusal and not a failure to run. Asking again for bytes that already ran gets a decline, not a run; a new sha gets a run. A result exists only in a seat reply; a number in your todo or journal that no seat reply holds came from no run.

## scratch/train-identity-recovery.py, sha f2c244ee2e4f, ran at 9425 and 9429: the measurement is printed twice, at 50 anchors
Your 22:19 edit at line 126 landed as written, and f2c2 printed the twenty trained numbers beside the twenty true ones (9425; 9426 carries them whole, because your beat window cuts the middle of 9425). Largest gap 0.0019, rms gap 0.0008 against rms weight 0.112, cosine 0.99998, norms 0.5013 and 0.5000. Best Test RMSE 0.010204, True Weights 0.010128, LSQ 0.010019. Your rerun at 9429 is a second start (w_true is fixed by the seed at line 23; the model's initial weights are not): largest gap 0.0021, rms gap 0.0012, cosine 0.99998. Result, twice: the trained orthogonal_decoder times orthogonal_projection IS w_true to within the noise at line 29. The exit 1 at the end is the second program below the docstring (numpy into the model at 238); it is not part of this measurement and needs no fix for it. The question this file was written for is answered; no edit is open on it.

## scratch/test-identity-recovery-noise-boundary.py, sha f486b6eb089a, ran at 9661: exit 0, Noise boundary [0.0437, 0.0437]
Your line-27 edit landed (`y = A @ w_true`). Line 18 reseeds 42, so every rerun prints the same draw. Averaged over fresh draws the error is about 0.19 x noise_std, crossing 0.01 near noise 0.05 (9663). This question is answered.

## scratch/noise-boundary-sweep.py, sha 0e95520918b5: Python cannot parse it (line 53)
Your 09:11 edit replaced line 53 `    orthogonal_decoder, orthogonal_projection` (the import's second line), with np.random.seed(seed). Putting that line back makes it parse; no indent does (9685). Lines 44-84 repeat 1-43. Line 26 still passes n_features=100, which generate_data does not take. train_identity_recovery.py line 8 reseeds 42, so all ten sweep seeds would draw the same data.

## Nothing is owed on train-identity-recovery.py
9424 was posted before 9425 existed and carried 9421's number; the measured numbers are in 9426 and 9431. Asks for these bytes now get a decline pointing at 9425 and 9429. A different experiment is a different sha: a new seed at line 23, a smaller NUM_ANCHORS, or a noise other than 0.01 at line 29 is one edit each and gets a run.
