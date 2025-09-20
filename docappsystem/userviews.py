from django.shortcuts import render, redirect, HttpResponse
from dasapp.models import DoctorReg, Specialization, CustomUser, Appointment, Page, PatientReg
from django.contrib.auth.decorators import login_required
from django.db.models import Q
import random
from datetime import datetime
from django.contrib import messages
import pandas as pd
from django.http import JsonResponse

def USERBASE(request):
    context = {}
    return render(request, "userbase.html", context)


def Index(request):
    doctorview = DoctorReg.objects.all().order_by('-id')
    doctorcount = DoctorReg.objects.all().count()
    patientcount = PatientReg.objects.all().count()
    page = Page.objects.all()

    context = {
        "doctorview": doctorview,
        "page": page,
        "doccount": doctorcount,
        "patcount": patientcount
    }
    return render(request, "index.html", context)

# @login_required(login_url='/')
def create_appointment(request):
    patient_admin = request.user
    if str(patient_admin) != 'AnonymousUser':
        doctorview = DoctorReg.objects.all()
        patient_instance = PatientReg.objects.get(admin = patient_admin.id)
        page = Page.objects.all()

        if request.method == "POST":
            appointmentnumber = random.randint(100000000, 999999999)
            fullname = request.POST.get("fullname")
            email = request.POST.get("email")
            mobilenumber = request.POST.get("mobilenumber")
            date_of_appointment = request.POST.get("date_of_appointment")
            # time_of_appointment = request.POST.get("time_of_appointment")
            doctor_id = request.POST.get("doctor_id")
            additional_msg = request.POST.get("additional_msg")

            # Retrieve the DoctorReg instance using the doctor_id
            doc_instance = DoctorReg.objects.get(id=doctor_id)

            # Validate that date_of_appointment is greater than today's date
            try:
                appointment_date = datetime.strptime(date_of_appointment, "%Y-%m-%d").date()
                today_date = datetime.now().date()

                if appointment_date <= today_date:
                    # If the appointment date is not in the future, display an error message
                    messages.error(
                        request, "Please select a date in the future for your appointment"
                    )
                    return redirect("appointment")  # Redirect back to the appointment page
            except ValueError:
                # Handle invalid date format error
                messages.error(request, "Invalid date format")
                return redirect("appointment")  # Redirect back to the appointment page

            # Create a new Appointment instance with the provided data
            appointmentdetails = Appointment.objects.create(
                patient = patient_instance,
                appointmentnumber=appointmentnumber,
                fullname=fullname,
                email=email,
                mobilenumber=mobilenumber,
                date_of_appointment=date_of_appointment,
                # time_of_appointment=time_of_appointment,
                doctor_id=doc_instance,
                additional_msg=additional_msg,
            )

            # Display a success message
            messages.success(
                request, "Your Appointment Request Has Been Sent. We Will Contact You Soon"
            )

            return redirect("appointment")

        context = {"doctorview": doctorview, "patient": patient_instance, "page": page, "user": patient_admin}
        return render(request, "appointment.html", context)
    else:
        messages.warning(request, "Please login to book appointment.")
        return redirect('login')


def User_Search_Appointments(request):
    page = Page.objects.all()

    if request.method == "GET":
        query = request.GET.get("query", "")
        if query:
            # Filter records where fullname or Appointment Number contains the query
            patient = Appointment.objects.filter(
                (Q(fullname__icontains=query) | Q(appointmentnumber__icontains=query)) & 
                ~Q(status='Completed')
            )
            messages.info(request, "Search against " + query)
            context = {"patient": patient, "query": query, "page": page}
            return render(request, "search-appointment.html", context)
        else:
            print("No Record Found")
            context = {"page": page}
            return render(request, "search-appointment.html", context)

    # If the request method is not GET
    context = {"page": page}
    return render(request, "search-appointment.html", context)


def View_Appointment_Details(request, id):
    page = Page.objects.all()
    patientdetails = Appointment.objects.filter(id=id)
    context = {"patientdetails": patientdetails, "page": page}

    return render(request, "user_appointment-details.html", context)
