#!/usr/bin/env python3
"""
Prediction Logger - Capture Model's Hallucinated User Responses as Training Data

When the model hallucinates a multi-turn dialogue, it's actually making a
prediction about what the user will say next. Instead of discarding this,
we capture it as training data:

Triplet Structure:
1. model_response (spoken) - What SAGE said to user
2. predicted_user_response (hallucinated) - What model thought user would say
3. actual_user_response (captured) - What user actually said

This creates a dataset of:
- Model predictions about conversation flow
- Reality vs prediction comparison
- Potential finetuning data for better dialogue modeling
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class PredictionLogger:
    """
    Logs model predictions about user responses (hallucinations) along with
    actual user responses for training data collection.
    """

    def __init__(self, log_dir: str = "~/sage_predictions"):
        """
        Initialize prediction logger.

        Args:
            log_dir: Directory to store prediction logs
        """
        self.log_dir = Path(log_dir).expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Current pending prediction (waiting for actual user response)
        self.pending_prediction: Optional[Dict[str, Any]] = None

        # Session log file
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = self.log_dir / f"predictions_{session_id}.jsonl"

        print(f"📊 Prediction logger initialized")
        print(f"   Log file: {self.session_file}")

    def capture_hallucination(
        self,
        model_response: str,
        predicted_user_response: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Capture a hallucinated user response (model's prediction).

        This is called when hallucination is detected. We save:
        1. What the model said (model_response)
        2. What the model predicted user would say (predicted_user_response)
        3. Wait for actual user response to complete the triplet

        Args:
            model_response: The assistant's response that was spoken
            predicted_user_response: The hallucinated user response
            context: Optional context (conversation history, metadata, etc.)
        """
        self.pending_prediction = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'model_response': model_response.strip(),
            'predicted_user_response': predicted_user_response.strip(),
            'context': context or {},
            'actual_user_response': None,  # To be filled when user speaks
            'status': 'pending'
        }

        print(f"\n📝 [PREDICTION] Captured hallucination:")
        print(f"   Model said: '{model_response[:60]}...'")
        print(f"   Model predicted user would say: '{predicted_user_response[:60]}...'")
        print(f"   Waiting for actual user response...")

    def log_actual_response(self, actual_user_response: str):
        """
        Log the actual user response, completing the prediction triplet.

        Args:
            actual_user_response: What the user actually said
        """
        if not self.pending_prediction:
            return  # No pending prediction to complete

        # Complete the triplet
        self.pending_prediction['actual_user_response'] = actual_user_response.strip()
        self.pending_prediction['status'] = 'complete'
        self.pending_prediction['response_time'] = time.time() - self.pending_prediction['timestamp']

        # Calculate similarity (simple word overlap for now)
        predicted_words = set(self.pending_prediction['predicted_user_response'].lower().split())
        actual_words = set(actual_user_response.lower().split())
        overlap = len(predicted_words & actual_words)
        total = len(predicted_words | actual_words)
        similarity = overlap / total if total > 0 else 0.0

        self.pending_prediction['similarity'] = similarity

        # Save to log file
        self._save_prediction(self.pending_prediction)

        print(f"\n✅ [PREDICTION] Triplet complete:")
        print(f"   Model predicted: '{self.pending_prediction['predicted_user_response'][:60]}...'")
        print(f"   User actually said: '{actual_user_response[:60]}...'")
        print(f"   Similarity: {similarity:.2%}")

        # Clear pending
        self.pending_prediction = None

    def _save_prediction(self, prediction: Dict[str, Any]):
        """Save prediction triplet to JSONL file."""
        with open(self.session_file, 'a') as f:
            f.write(json.dumps(prediction) + '\n')

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about collected predictions."""
        if not self.session_file.exists():
            return {
                'total_predictions': 0,
                'avg_similarity': 0.0,
                'pending': 0
            }

        predictions = []
        with open(self.session_file, 'r') as f:
            for line in f:
                predictions.append(json.loads(line))

        complete = [p for p in predictions if p['status'] == 'complete']
        avg_similarity = sum(p['similarity'] for p in complete) / len(complete) if complete else 0.0

        return {
            'total_predictions': len(predictions),
            'complete_predictions': len(complete),
            'avg_similarity': avg_similarity,
            'pending': 1 if self.pending_prediction else 0
        }

    def export_training_data(self, output_file: Optional[str] = None) -> str:
        """
        Export predictions in a format suitable for finetuning.

        Returns path to exported file.
        """
        if output_file is None:
            output_file = self.log_dir / f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            output_file = Path(output_file)

        if not self.session_file.exists():
            print("No predictions to export")
            return None

        # Read all predictions
        predictions = []
        with open(self.session_file, 'r') as f:
            for line in f:
                predictions.append(json.loads(line))

        # Filter to complete predictions only
        complete = [p for p in predictions if p['status'] == 'complete']

        # Format for training
        training_data = {
            'metadata': {
                'created': datetime.now().isoformat(),
                'total_predictions': len(complete),
                'avg_similarity': sum(p['similarity'] for p in complete) / len(complete) if complete else 0.0
            },
            'examples': [
                {
                    'assistant_response': p['model_response'],
                    'predicted_user_response': p['predicted_user_response'],
                    'actual_user_response': p['actual_user_response'],
                    'similarity': p['similarity'],
                    'context': p.get('context', {})
                }
                for p in complete
            ]
        }

        with open(output_file, 'w') as f:
            json.dump(training_data, f, indent=2)

        print(f"📦 Exported {len(complete)} prediction triplets to: {output_file}")
        return str(output_file)
