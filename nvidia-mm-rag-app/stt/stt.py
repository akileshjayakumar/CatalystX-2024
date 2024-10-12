import base64
import streamlit as st
import os
import speech_recognition as sr


recognizer = sr.Recognizer()

def speech_to_text(audio_data):
    with open(audio_data, "rb") as audio_file:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                text = "No speech detected."
    return text
