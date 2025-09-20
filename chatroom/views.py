import os
import re
from django.urls import reverse
import nltk
import joblib
import random
import difflib
import pandas as pd
from nltk.tree import Tree
from datetime import datetime, timedelta
from fuzzywuzzy import process
from dateutil.parser import parse
from nltk import ne_chunk, pos_tag
from collections import defaultdict
from django.shortcuts import get_object_or_404, render
from django.http import JsonResponse
from nltk import word_tokenize, pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet as wn
from django.utils.timezone import make_aware, now
from nltk.sentiment import SentimentIntensityAnalyzer
from django.contrib.auth.decorators import login_required
from dasapp.models import DoctorReg, Specialization, Appointment, CustomUser, PatientReg
from dateparser import parse as dateparser_parse
import parsedatetime
import spacy

# Load model and data files only once (outside view functions for efficiency)
current_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(current_dir, "RandomForestModel.pkl"))
symptom_list = joblib.load(os.path.join(current_dir, "SymptomsList.pkl"))
co_occurrence_graph = joblib.load(os.path.join(current_dir, "CooccuranceGraph.pkl"))
disease_symptom_mapping = joblib.load(
    os.path.join(current_dir, "DiseaseSymptomMapping.pkl")
)
disease_description_df = pd.read_csv(
    os.path.join(current_dir, "disease_description.csv")
)
disease_precaution_df = pd.read_csv(os.path.join(current_dir, "disease_precaution.csv"))
disease_specialization_df = pd.read_csv(
    os.path.join(current_dir, "disease_specializations.csv")
)

# Initialize state dictionary outside the view
USER_STATES = {}

nltk.download("punkt")
nltk.download("words")
nltk.download("stopwords")
nltk.download("vader_lexicon")
nltk.download("maxent_ne_chunker")
nltk.download("averaged_perceptron_tagger")
nltk.download("en_core_web_sm")

# Global initialization of spaCy and parsedatetime
nlp = spacy.load("en_core_web_sm")
cal = parsedatetime.Calendar()


def reset_user_state(user_id):
    USER_STATES[user_id] = {
        "stage": "initial_greeting",
        "confirmed_symptoms": [],
        "initial_symptoms": [],
        "remaining_symptoms": symptom_list.copy(),
        "positive_response_count": 0,
        "doctor_name": "",
        "available_doctors": [],
        "next_symptom": None,
        "response_category": [],
        "user_info": {},
        "appointment_info": {},
    }


def extract_relevant_symptoms(user_input, symptom_list):
    """
    Extract relevant symptoms from user input using NLTK and approximate matching.
    """
    # Load stop words
    stop_words = set(stopwords.words("english"))

    # Tokenize and filter user input
    tokens = word_tokenize(user_input)
    filtered_tokens = [
        word.lower()
        for word in tokens
        if word.isalnum() and word.lower() not in stop_words
    ]

    # Find closest matches for each token from the symptom list
    relevant_symptoms = []
    for token in filtered_tokens:
        match, score = process.extractOne(token, symptom_list)
        if score > 80:  # Set a threshold for similarity (80% confidence)
            relevant_symptoms.append(match)

    # Return unique symptoms (avoiding duplicates)
    return list(set(relevant_symptoms))


def get_relevant_diseases(confirmed_symptoms, disease_symptom_mapping):
    """
    This function identifies diseases related to the user's confirmed symptoms.
    It returns a dictionary where diseases are keys and the number of confirmed symptoms that match is the value.
    """
    disease_match_count = defaultdict(int)

    for disease, symptoms in disease_symptom_mapping.items():
        for symptom in confirmed_symptoms:
            if symptom in symptoms:
                disease_match_count[disease] += 1

    # Sort diseases based on the number of confirmed symptoms matched (highest first)
    sorted_diseases = sorted(
        disease_match_count.items(), key=lambda x: x[1], reverse=True
    )

    return sorted_diseases


def get_next_symptom_to_confirm(
    confirmed_symptoms, remaining_symptoms, disease_symptom_mapping, co_occurrence_graph
):
    """
    This function determines the next symptom to ask the user about, based on the best-fit disease.
    """
    # Step 1: Get diseases related to the confirmed symptoms
    sorted_diseases = get_relevant_diseases(confirmed_symptoms, disease_symptom_mapping)

    # Step 2: For each disease, check which symptoms are related to it and available for questioning
    for disease, match_count in sorted_diseases:
        # Get the symptoms for the current disease that are still in the remaining symptoms
        relevant_symptoms = [
            symptom
            for symptom in disease_symptom_mapping[disease]
            if symptom in remaining_symptoms
        ]

        # Step 3: Prioritize symptoms that are highly co-occurring with confirmed symptoms
        if relevant_symptoms:
            weighted_symptoms = defaultdict(float)

            # Adjust symptom weights based on co-occurrence with confirmed symptoms
            for confirmed in confirmed_symptoms:
                if confirmed in co_occurrence_graph:
                    for related_symptom, weight in co_occurrence_graph[confirmed]:
                        if related_symptom in relevant_symptoms:
                            weighted_symptoms[related_symptom] += weight

            # Sort symptoms based on co-occurrence weight (highest first)
            sorted_relevant_symptoms = sorted(
                weighted_symptoms.items(), key=lambda x: x[1], reverse=True
            )

            # Return the next symptom to confirm (highest weight)
            if sorted_relevant_symptoms:
                return sorted_relevant_symptoms[0][
                    0
                ]  # Return the symptom with the highest weight

    # If no relevant symptoms found, return None (indicating no symptoms to ask)
    return None


# Initialize Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()


def classify_response(response):
    """
    Classify user response into affirmative, negative, or uncertain.
    """
    # Common keywords
    affirmative_keywords = {
        "yes",
        "yeah",
        "yep",
        "sure",
        "of course",
        "totally",
        "indeed",
        "right",
        "affirmative",
        "aye",
        "agreed",
        "alright",
        "true",
        "that's correct",
        "sounds good",
        "you bet",
        "i have",
    }

    negative_keywords = {
        "no",
        "nope",
        "nah",
        "not at all",
        "no way",
        "never",
        "negative",
        "absolutely not",
        "i don't think so",
        "not really",
        "no chance",
        "not in the slightest",
        "certainly not",
        "not happening",
        "don’t think so",
        "i refuse",
    }

    uncertain_keywords = {
        "maybe",
        "perhaps",
        "not sure",
        "don’t know",
        "kind of",
        "sort of",
        "i guess",
        "it depends",
        "i think so",
        "i’m unsure",
        "probably",
        "possibly",
        "not really",
        "could be",
        "might be",
    }

    # Keyword-based classification
    if response.lower() in affirmative_keywords:
        return "affirmative"
    elif response.lower() in negative_keywords:
        return "negative"
    elif response.lower() in uncertain_keywords:
        return "uncertain"

    # Fallback to sentiment analysis
    sentiment = sia.polarity_scores(response)
    polarity = sentiment["compound"]

    if polarity > 0.2:
        return "affirmative"
    elif polarity < -0.2:
        return "negative"
    else:
        return "uncertain"


def is_user_confused(user_input):
    # List of confusion-related keywords and multi-word phrases
    confused_synonyms = [
        "confused",
        "unclear",
        "perplexed",
        "bewildered",
        "lost",
        "baffled",
        "rephrase",
        "didn't understand",
        "don't understand",
        "did not understand",
        "do not understand",
        "don't get it",
        "can't follow",
        "don't know",
        "what is",
        "?",
        "what",
    ]

    # Check if the user input exactly matches any of the confusion-related phrases
    for phrase in confused_synonyms:
        if phrase.lower() in user_input.lower():
            return True

    # Tokenize the response and find synonyms in WordNet for individual words
    tokens = word_tokenize(user_input.lower())
    for token in tokens:
        for synonym in confused_synonyms:
            synsets = wn.synsets(token)
            for syn in synsets:
                if any(lemma.name() == synonym for lemma in syn.lemmas()):
                    return True
    return False


def get_symptom_definition(symptom):
    synsets = wn.synsets(symptom)
    if synsets:
        # Use the first synset's definition
        return synsets[0].definition()
    else:
        return "I couldn't find a definition for that symptom. You can say not sure if confused."


def predict_disease(symptoms, symptom_list, model):
    """
    Predict the disease based on confirmed symptoms.
    """
    symptom_vector = [1 if symptom in symptoms else 0 for symptom in symptom_list]
    input_df = pd.DataFrame([symptom_vector], columns=symptom_list)
    predicted_disease = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)
    confidence = max(probabilities[0])
    return predicted_disease, confidence


def get_disease_info(disease):
    description_row = disease_description_df[
        disease_description_df["Disease"].str.lower() == disease.lower()
    ]
    description = (
        description_row["Description"].values[0]
        if not description_row.empty
        else "No description available."
    )
    precaution_row = disease_precaution_df[
        disease_precaution_df["Disease"].str.lower() == disease.lower()
    ]
    precautions = (
        [
            precaution_row[f"Precaution_{i+1}"].values[0]
            for i in range(4)
            if f"Precaution_{i+1}" in precaution_row.columns
            and pd.notna(precaution_row[f"Precaution_{i+1}"].values[0])
        ]
        if not precaution_row.empty
        else ["No precautions available."]
    )
    return description, precautions


def normalize_name(name):
    """
    Normalizes a name by capitalizing it in 'Firstname Lastname' format.
    Adds 'Dr.' prefix if not present.
    """
    name_parts = name.split()
    if len(name_parts) < 2:  # Ensure the name has at least two parts
        return None  # Invalid name format
    normalized = " ".join([part.capitalize() for part in name_parts])
    if not normalized.startswith("Dr."):
        normalized = f"Dr. {normalized}"
    return normalized


def extract_and_format_name(user_response, name_list):
    """
    Extracts and formats the name from user input using NLTK and matches it with a list of known names.
    """
    # Tokenize and tag user response
    capitalized_response = (
        user_response.title()
    )  # Capitalize each word for better tagging
    tokens = word_tokenize(capitalized_response)
    tagged = pos_tag(tokens)

    # List of common nouns to exclude
    common_nouns = {
        "appointment",
        "meeting",
        "doctor",
        "schedule",
        "time",
        "clinic",
        "hospital",
        "availability",
        "consultation",
        "session",
        "checkup",
        "service",
        "health",
        "date",
        "day",
        "morning",
        "afternoon",
        "evening",
        "assistant",
        "staff",
        "office",
        "choose",
    }

    # Extract consecutive proper nouns (NNP) or related tags as potential names
    potential_names = []
    current_name = []
    for word, tag in tagged:
        if (
            tag in ("NNP", "NN", "JJ") and word.lower() not in common_nouns
        ):  # Proper noun, noun, or adjective
            current_name.append(word)
        elif current_name:  # End of a potential name
            potential_names.append(" ".join(current_name))
            current_name = []
    if current_name:  # Append any remaining name
        potential_names.append(" ".join(current_name))

    for potential_name in potential_names:
        if len(potential_name.split()) < 2:
            return None, "FullNameNotFound"

    # Normalize names in the name_list
    normalized_list = [
        normalize_name(name) for name in name_list if normalize_name(name)
    ]
    # print(normalized_list)

    # Normalize and match potential names
    for potential_name in potential_names:
        normalized_response = normalize_name(potential_name)
        if not normalized_response:
            continue
        extracted_name = difflib.get_close_matches(
            normalized_response, normalized_list, n=1, cutoff=0.5
        )
        if extracted_name:
            if extracted_name[0] != normalized_response:
                return (
                    extracted_name[0],
                    f"You said {normalized_response}. i beleive you meaned {extracted_name[0]} and we can proceed.",
                )
            else:
                return extracted_name[0], "Success"
    return None, "DoctorNotFound"
    
def extract_date(sentence):
    """
    Extract date from a given sentence using various methods.
    Returns a datetime.date object if a date is found, else None.
    """
    # Extraction methods in order of preference
    extraction_methods = [
        extract_with_parsedatetime,
        extract_with_dateparser,
        extract_with_regex,
        extract_with_spacy
    ]

    # Try each method
    for method in extraction_methods:
        date = method(sentence)
        if date:
            # Ensure only the date part is returned
            return date.date()

    return None

def extract_with_parsedatetime(sentence):
    """
    Extract date using parsedatetime.
    """
    try:
        time_struct, success = cal.parse(sentence)
        if success:
            parsed_date = datetime(*time_struct[:6])
            return parsed_date
    except Exception as e:
        print(f"Parsedatetime error: {e}")
    
    return None


def extract_with_dateparser(sentence):
    """
    Extract date using dateparser.date_of_apt
    """
    try:
        parsed_date = dateparser_parse(
            sentence, 
            settings={'RELATIVE_BASE': datetime.now()}
        )
        if parsed_date:
            return parsed_date
    except Exception as e:
        print(f"Dateparser error: {e}")
    
    return None


def extract_with_regex(sentence):
    """
    Extract date using regex patterns.
    """
    date_patterns = [
        r'\b(\d{1,2}/\d{1,2}/\d{2,4})\b',  # MM/DD/YYYY or DD/MM/YYYY
        r'\b(\d{4}-\d{1,2}-\d{1,2})\b',    # YYYY-MM-DD
        r'\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b',
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b'
    ]

    for pattern in date_patterns:
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match:
            try:
                parsed_date = dateparser_parse(match.group(1))
                if parsed_date:
                    return parsed_date
            except Exception:
                pass
    
    return None


def extract_with_spacy(sentence):
    """
    Extract date using spaCy's named entity recognition.
    """
    doc = nlp(sentence)
    for ent in doc.ents:
        if ent.label_ in ['DATE', 'TIME']:
            try:
                parsed_date = dateparser_parse(ent.text)
                if parsed_date:
                    return parsed_date
            except Exception:
                pass
    
    return None


def validate_date(parsed_date):
    """
    Validates if the parsed date is in the future.
    """
    current_time = now().date()
    if parsed_date <= current_time:
        return False, "Provide a future date for the appointment."
    return True, None

def symptoms_to_sentence(symptoms):
    if not symptoms:
        return "I have no symptoms."
    if len(symptoms) == 1:
        return f"I have {symptoms[0]}."
    if len(symptoms) == 2:
        return f"I have {symptoms[0]} and {symptoms[1]}."
    return f"I have {', '.join(symptoms[:-1])}, and {symptoms[-1]}."


def book_appointment(user_id, current_user_id):
    user_state = USER_STATES[user_id]
    try:
        doctor_name = user_state["doctor_name"]
        name_parts = doctor_name.split()
        first_name = " ".join(name_parts[:2])
        last_name = name_parts[2]
        doc_inst = DoctorReg.objects.get(
            admin__first_name=first_name, admin__last_name=last_name
        )
        appointment_number = random.randint(100000000, 999999999)

        current_user = get_object_or_404(CustomUser, id=current_user_id)
        patient = PatientReg.objects.get(admin=current_user)
        patient_fullname = f"{current_user.first_name} {current_user.last_name}"

        date_of_apt = user_state["appointment_info"]["date"]
            
        # Check if the user already has an appointment within the last 15 days
        fifteen_days_ago = date_of_apt - timedelta(days=15)
        previous_appointment = Appointment.objects.filter(
            patient=patient,
            doctor_id=doc_inst,
            date_of_appointment__gte=fifteen_days_ago,
            date_of_appointment__lt=date_of_apt,
            status__in=["PaymentDone", "Completed"]
        ).first()
            
        apt_count = Appointment.objects.filter(date_of_appointment=date_of_apt, doctor_id=doc_inst).count()
        if apt_count < doc_inst.dalily_patients:
            Appointment.objects.create(
                patient=patient,
                appointmentnumber=appointment_number,
                fullname=patient_fullname,
                mobilenumber=patient.mobilenumber,
                email=current_user.email,
                date_of_appointment=date_of_apt,
                doctor_id=doc_inst,
                additional_msg=user_state["appointment_info"].get("additional_info", "nil"),
            )
            # Return FreeBooking if a previous appointment exists within 15 days
            if previous_appointment:
                return ("FreeBooking", appointment_number)
            return ("success", appointment_number)
        else:
            return("ApmtFull", 0)
    except Exception as e:
        return str(e), 0


@login_required(login_url="/login")
def chat(request):
    user_id = request.session.session_key
    if not user_id:
        request.session.create()
        user_id = request.session.session_key

    if not user_id or user_id not in USER_STATES:
        reset_user_state(user_id)
    user_state = USER_STATES[user_id]

    if request.method == "GET":
        reset_user_state(user_id)
        return render(request, "chat.html")
    elif request.method == "POST":
        user_message = request.POST.get("message").strip().lower()
        response_data = {"bot_message": ""}
        thank_words = [
            "Thanks",
            "Thank you",
            "Thanks a lot",
            "Many thanks",
            "Thank you very much",
            "Thanks so much",
            "I appreciate it",
            "I appreciate you",
            "Much appreciated",
            "I'm grateful",
            "I'm thankful",
            "Grateful thanks",
            "Thanks a million",
            "I owe you one",
            "Much obliged",
            "I can't thank you enough",
            "Thanks heaps",
            "Sincere thanks",
            "Endless thanks",
            "I’m truly grateful",
            "I’m thankful for this",
            "Big thanks",
            "Deeply appreciated",
            "Thanks from the bottom of my heart",
            "I’m much obliged"
        ]

        # Handle greeting
        if user_state["stage"] == "initial_greeting":
            user_msg = difflib.get_close_matches(
                user_message, ["hi", "hello", "hey", "hai"], n=1, cutoff=0.5
            )
            grateful = difflib.get_close_matches(
                user_message, thank_words, n=1, cutoff=0.5
            )
            if user_msg:
                response_data["bot_message"] = (
                    "Hello! Welcome to the disease prediction chatbot. Do you experience any symptoms?"
                )
                user_state["stage"] = "awaiting_symptom_input"
                return JsonResponse(response_data)
            elif grateful:
                response_data["bot_message"] = (
                    "You're welcome! If you need any more help, feel free to ask. 😊"
                )
                return JsonResponse(response_data)
            else:
                response_data["bot_message"] = (
                    "Do you experience any symptoms ?"
                )
                user_state["stage"] = "awaiting_symptom_input"
                return JsonResponse(response_data)

        # Symptom input and disease prediction logic
        if user_state["stage"] == "awaiting_symptom_input":
            response = classify_response(user_message)
            if response == "affirmative":
                response_data["bot_message"] = "Please enter your symptoms."
                user_state["stage"] = "collecting_symptoms"
                return JsonResponse(response_data)
            elif response == "negative":
                response_data["bot_message"] = (
                    "Alright, take care! Let me know if you feel something."
                )
                reset_user_state(user_id)
                return JsonResponse(response_data)

        elif user_state["stage"] == "collecting_symptoms":
            if not user_state["confirmed_symptoms"]:
                # Step 1: Handle initial symptoms
                initial_symptoms = extract_relevant_symptoms(user_message, symptom_list)
                user_state["initial_symptoms"] = initial_symptoms
                if initial_symptoms:
                    user_state["confirmed_symptoms"].extend(initial_symptoms)
                    user_state["remaining_symptoms"] = [
                        symptom
                        for symptom in user_state["remaining_symptoms"]
                        if symptom not in user_state["confirmed_symptoms"]
                    ]
                    response_data["bot_message"] = (
                        f"Ok I understood, Thanks for sharing.<br>"
                    )
                else:
                    response_data["bot_message"] = (
                        "I couldn't recognize any symptoms in your message. Could you rephrase or specify your symptoms?"
                    )
                    return JsonResponse(response_data)

            # Step 2: Handle a response to a previously asked symptom
            if user_state["next_symptom"]:
                next_symptom = user_state["next_symptom"]
                if is_user_confused(user_message):
                    definition = get_symptom_definition(next_symptom)
                    response_data["bot_message"] = f"{next_symptom}: {definition}"
                    response_data[
                        "bot_message"
                    ] += f"<br><br>Do you have {next_symptom}?"
                    return JsonResponse(response_data)

                response_category = classify_response(user_message)

                if response_category == "affirmative":
                    user_state["confirmed_symptoms"].append(next_symptom)
                    user_state["positive_response_count"] += 1
                    response_data["bot_message"] = "Ok. "
                elif response_category == "negative":
                    response_data["bot_message"] = (
                        f"Ok then we can skip {next_symptom} <br>"
                    )
                elif response_category == "uncertain":
                    response_data["bot_message"] = (
                        f"Noted: {next_symptom} might be a possibility, but we'll confirm with related symptoms. <br><br>"
                    )
                else:
                    response_data["bot_message"] = (
                        "I didn't understand that. Please respond with yes, no, or something similar."
                    )
                    response_data[
                        "bot_message"
                    ] += f"<br><br>Do you have {next_symptom}?"
                    return JsonResponse(response_data)

                user_state["next_symptom"] = None
                predicted_disease, confidence = predict_disease(
                    user_state["confirmed_symptoms"], symptom_list, model
                )
                user_state["last_prediction"] = predicted_disease
                user_state["prediction_confidence"] = confidence

                if confidence >= 0.8 or user_state["positive_response_count"] > 6:
                    if confidence > 0.2:
                        description, precautions = get_disease_info(predicted_disease)
                        disclaimer = (
                            f"I'm not really confident about this prediction. I'd say {int(confidence * 100)}% confident."
                            if confidence < 0.8
                            else ""
                        )
                        specialization = disease_specialization_df.loc[
                            disease_specialization_df["Disease"] == predicted_disease,
                            "Specialization",
                        ].values
                        specialization = (
                            specialization[0] if len(specialization) > 0 else "unknown"
                        )
                        user_state["current_specs"] = specialization

                        response_data["bot_message"] = (
                            f"According to the symptoms, you may have {predicted_disease}.<br>"
                            f"<br>{description}<br><br>Here's some precautions you could take:<br>"
                            + " ".join(
                                [f"<br>- {precaution}." for precaution in precautions]
                            )
                            + (f"<br><br>{disclaimer}" if disclaimer else "")
                            + f"<br><br>It is recommended that you see a {specialization} for more help.<br><br>Would you like to book an appointment ?"
                        )
                        user_state["next_symptom"] = None
                        user_state["stage"] = "awaiting_appointment_confirmation"
                        return JsonResponse(response_data)
                    else:
                        pred_disease, conf = predict_disease(
                            user_state["initial_symptoms"], symptom_list, model
                        )
                        description, precautions = get_disease_info(pred_disease)
                        disclaimer = f"Remember this is just my guess. I would say I am only {int(conf * 100)}% confident."
                        specialization = disease_specialization_df.loc[
                            disease_specialization_df["Disease"] == predicted_disease,
                            "Specialization",
                        ].values
                        specialization = (
                            specialization[0] if len(specialization) > 0 else "unknown"
                        )
                        user_state["current_specs"] = specialization

                        response_data["bot_message"] = (
                            f"Based on the symptoms you said, I couldn't came to a conclusion. But i think you may have {predicted_disease}.<br>"
                            f"<br>{description}<br><br>Here's some precautions you could take.<br>"
                            + " ".join(
                                [f"<br>- {precaution}." for precaution in precautions]
                            )
                            + (f"<br><br>{disclaimer}")
                            + f"<br><br>It is recommended that you visit a {specialization} for further assistance.<br><br>Would you like to book an appointment?"
                        )
                        user_state["next_symptom"] = None
                        user_state["stage"] = "awaiting_appointment_confirmation"
                        return JsonResponse(response_data)

            next_symptom = get_next_symptom_to_confirm(
                user_state["confirmed_symptoms"],
                user_state["remaining_symptoms"],
                disease_symptom_mapping,
                co_occurrence_graph,
            )

            if next_symptom:
                user_state["next_symptom"] = next_symptom
                user_state["remaining_symptoms"].remove(next_symptom)
                response_data["bot_message"] += f" Do you have {next_symptom} ?"
            else:
                response_data["bot_message"] = (
                    "I'm out of related symptoms to ask. Based on the information provided, I'll proceed with the prediction."
                )
                return JsonResponse(response_data)

            return JsonResponse(response_data)

        elif user_state["stage"] == "awaiting_appointment_confirmation":
            response = classify_response(user_message)
            if response == "affirmative":
                spec_inst = Specialization.objects.get(
                    sname=user_state["current_specs"]
                )
                doctors = DoctorReg.objects.filter(specialization_id=spec_inst.id)
                for doctor in doctors:
                    user_state["available_doctors"].append(
                        f"{doctor.admin.first_name} {doctor.admin.last_name}"
                    )
                response_data["bot_message"] = (
                    "Here is the list of doctors we have. Please select a doctor:<br>"
                    + "<br>".join(
                        [
                            f"- {doc.admin.first_name} {doc.admin.last_name}"
                            for doc in doctors
                        ]
                    )
                )
                user_state["stage"] = "selecting_doctor"
                return JsonResponse(response_data)
            elif response == "negative":
                response_data["bot_message"] = (
                    "Alright, take care! Feel free to reach out if you change your mind."
                )
                reset_user_state(user_id)
                return JsonResponse(response_data)

        elif user_state["stage"] == "selecting_doctor":
            formatted_name, message = extract_and_format_name(
                user_message, user_state["available_doctors"]
            )
            if message and message == "DoctorNotFound":
                spec_inst = Specialization.objects.get(
                    sname=user_state["current_specs"]
                )
                doctors = DoctorReg.objects.filter(specialization_id=spec_inst.id)
                response_data["bot_message"] = (
                    "I think you entered a wrong doctor name. Please select a doctor from the provided list."
                )
                response_data[
                    "bot_message"
                ] += "<br>Here is the list of doctors we have,<br>" + "<br>".join(
                    [
                        f"- {doc.admin.first_name} {doc.admin.last_name}"
                        for doc in doctors
                    ]
                )
                return JsonResponse(response_data)
            elif message and message == "FullNameNotFound":
                response_data["bot_message"] = (
                    "Please provide the full name of doctor, since we have multiple doctors with same name."
                )
                return JsonResponse(response_data)
            elif message and message != "Success":
                response_data["bot_message"] = f"{message}<br>"

            user_state["doctor_name"] = formatted_name
            response_data[
                "bot_message"
            ] += "Please provide a date for your appointment."
            user_state["stage"] = "awaiting_doa"
            return JsonResponse(response_data)

        elif user_state["stage"] == "awaiting_doa":
            parsed_date = extract_date(user_message)

            if parsed_date:
                # Validate if the date is in the future
                is_valid, message = validate_date(parsed_date)
                if not is_valid:
                    response_data["bot_message"] = message
                    return JsonResponse(response_data)

                # Booking logic
                current_user = request.user.id
                add_info = symptoms_to_sentence(user_state["confirmed_symptoms"])
                user_state["appointment_info"] = {
                    "date": parsed_date,
                    "additional_info": add_info,
                }
                status, app_id = book_appointment(user_id, current_user)
                if status == "success":
                    response_data["bot_message"] = (
                        f"Appointment scheduled. Use this reference number {app_id} to check the status of your appointment.<br>"
                        "Once the doctor has approved it, you can pay the fee and visit the doctor at the time allotted to you."
                    )
                elif status == "FreeBooking":
                    response_data["bot_message"] = (
                        f"Appointment scheduled. Use this reference number {app_id} to check the status of your appointment.<br>"
                        "You don't need to pay the fee because you are scheduling an appointment with the same doctor within 15 days of your last visit. Once the doctor has approved it, you may visit the doctor at the time alloted for you."
                    )
                elif status == "ApmtFull":
                    response_data["bot_message"] = (
                        f"Sorry bookings are closed for the provided date ! Please choose another date."
                    )
                    return JsonResponse(response_data)
                elif status == "InvalidDate":
                    # Invalid date case
                    response_data["bot_message"] = (
                        f"I failed to find a valid date in your message. Please provide a specific date to proceed. date: {app_id}"
                    )
                    return JsonResponse(response_data)
                else:
                    response_data["bot_message"] = (
                        f"Unable to book appointment. Please try again. {status}"
                    )
                reset_user_state(user_id)
                return JsonResponse(response_data)
            else:
                # Invalid date case
                response_data["bot_message"] = (
                    "I couldn't find a valid date in your message. Please provide a specific date to proceed."
                )
                return JsonResponse(response_data)

        # Default response
        response_data["bot_message"] = (
            f"Sorry I didn't understand, I am still learning. Can you rephrase that ?"
        )
        return JsonResponse(response_data)

