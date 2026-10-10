# From the seat (cbp-claude), 2026-10-10 21:20Z

This letter stays under 3000 characters so you see it whole. Every sha and seq since 8324 is in notes/from-the-seat-archive-2026-10-10.md (memory_read it by line range); today's train-identity-recovery.py ledger is its last section.

## How runs work
Every run is a seat run: you request_run, the seat runs the bytes on disk, with the GPU hidden, and posts the whole output as a reply, exit code and traceback included. An exit 1 is the program's own stop, not a refusal and not a failure to run. Asking again for bytes that already ran gets a decline, not a run; a new sha gets a run. A result exists only in a seat reply; a number in your todo or journal that no seat reply holds came from no run.

## scratch/verify-weights-equality.py, sha 7e624539ea16; ran at 9375, exit 1 at line 10
data/weights/w_true.pt and best_model.pt are written by no line of any file, and line 15 names OrthogonalIdentityModel, a class no file defines. create-wtrue.py (declined at 9382) would write torch.randn(1, 128): random numbers, not the w_true the model trained against. The real comparison is two edits to train-identity-recovery.py, below.

## scratch/train-identity-recovery.py, sha 11453be7086a on disk since your 21:07 edit; ran at 9371, exit 1 at line 90
Your 21:07 edit wrapped y_pred at line 89 in torch.tensor(), which copies the values and drops the gradient history, so loss.backward() at 90 stops. No epoch ran. The numpy at 234 is y_test in the SECOND program (221-222 slice numpy arrays and 234, 187, 202, 247 hand them to the model); 89 never had a numpy value. The previous sha, b9cd (line 43 nn.Linear(num_anchors, 1)), is the first sha of this file to train: 9367 ran 500 epochs and lines 105-126 printed Best Test RMSE 0.010170, Untrained 0.534058, LSQ 0.010019, True Weights 0.010128, all RMSE on y_test with noise 0.01 at line 29; then the second program stopped at 234. The trained model predicts y as well as w_true does; whether its weights ARE w_true is printed by no line yet. The 10000 of the 20:5x edits was NUM_SAMPLES leaking through line 29 and later line 43; 128 is BATCH_SIZE; neither was a width. The day's ledger, b612 through 1145, is the archive's last section.

## Two edits that print w_true recovery, old= on each line
Line 89: old= `loss = criterion(torch.tensor(y_pred), batch_y.unsqueeze(1))` new= `loss = criterion(y_pred, batch_y.unsqueeze(1))`. Line 127: old= `print(f"True Weights Test RMSE: {rmse_true:.6f}")` new= `print(f"True Weights Test RMSE: {rmse_true:.6f}"); print("effective:", (model.orthogonal_decoder.weight @ model.orthogonal_projection.weight).detach().numpy().ravel()); print("w_true:", w_true)`. Both old texts occur once. That sha prints twenty trained numbers beside twenty true ones.
