import numpy as np
import logging

class DiseasePredictor:
    def __init__(self, model, symptom_graph):
        self.model = model
        self.symptom_graph = symptom_graph
        self.confirmed_symptoms = set()
        self.predicted_disease = None
        self.prediction_confidence = 0.0
        logging.basicConfig(level=logging.DEBUG)

    def update_confirmed_symptoms(self, symptoms):
        for symptom in symptoms:
            if symptom in self.symptom_graph:
                self.confirmed_symptoms.add(symptom)
                logging.debug(f"Added confirmed symptom: {symptom}")
            else:
                logging.warning(f"Symptom not recognized: {symptom}")
        logging.debug(f"Updated confirmed symptoms: {self.confirmed_symptoms}")

    def suggest_next_symptom(self):
        for symptom in self.confirmed_symptoms:
            related_symptoms = self.symptom_graph.get(symptom, [])
            for related_symptom in related_symptoms:
                if related_symptom not in self.confirmed_symptoms:
                    logging.debug(f"Suggested next symptom: {related_symptom}")
                    return related_symptom
        logging.debug("No more symptoms to suggest.")
        return None

    def predict_disease(self):
        if not self.confirmed_symptoms:
            logging.error("No confirmed symptoms to predict disease.")
            return None, 0.0

        try:
            symptom_vector = self._symptoms_to_vector(self.confirmed_symptoms)
            logging.debug(f"Symptom vector: {symptom_vector}")
            prediction = self.model.predict([symptom_vector])[0]
            confidence = np.max(self.model.predict_proba([symptom_vector])[0])
            
            self.predicted_disease = prediction
            self.prediction_confidence = confidence
            
            logging.debug(f"Predicted disease: {prediction} with confidence: {confidence}")
            return prediction, confidence
        except Exception as e:
            logging.error(f"Error during disease prediction: {e}", exc_info=True)
            return None, 0.0

    def _symptoms_to_vector(self, symptoms):
        symptom_list = sorted(self.symptom_graph.keys())
        return [1 if symptom in symptoms else 0 for symptom in symptom_list]

    def get_disease_info(self):
        # Mock disease information retrieval
        description = f"{self.predicted_disease} is a condition..."
        precautions = ["Stay hydrated", "Rest", "Consult a healthcare provider"]
        return description, precautions
