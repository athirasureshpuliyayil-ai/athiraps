from django.shortcuts import get_object_or_404, render,redirect,HttpResponse
from django.contrib.auth.decorators import login_required
from dasapp.models import DoctorReg,Specialization,CustomUser,Appointment, PatientReg
from django.contrib import messages
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from datetime import datetime

def PatientSignup(request):
    if request.method == "POST":
        pic = request.FILES.get('pic')
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        username = request.POST.get('username')
        email = request.POST.get('email')
        mobno = request.POST.get('mobno')
        password = request.POST.get('password')

        if CustomUser.objects.filter(email=email).exists():
            messages.warning(request,'Email already exist')
            return redirect('patsignup')
        if CustomUser.objects.filter(username=username).exists():
            messages.warning(request,'Username already exist')
            return redirect('patsignup')
        else:
            user = CustomUser(
               first_name=first_name,
               last_name=last_name,
               username=username,
               email=email,
               user_type=3,
               profile_pic = pic,
            )
            user.set_password(password)
            user.save()
            patient = PatientReg(
                admin = user,
                mobilenumber = mobno,
                
            )
            patient.save()            
            messages.success(request,'Signup Successfully')
            return redirect('patsignup')

    return render(request,'pat/patreg.html')

@login_required(login_url='/')
def PatientHOME(request):
    patient_admin = request.user
    patient_inst = PatientReg.objects.get(admin = patient_admin.id)
    allaptcount = Appointment.objects.filter(patient=patient_inst).count
    context = {
        'allaptcount':allaptcount
    }
    return render(request, 'pat/pathome.html',context)

@login_required(login_url='/')
def ViewAppointments(request):
    patient_admin = request.user
    patient_inst = PatientReg.objects.get(admin = patient_admin.id)
    apps = Appointment.objects.filter(patient = patient_inst).order_by('-pk')
    # Pagination
    paginator = Paginator(apps, 5)  # Show 10 appointments per page
    page = request.GET.get('page')
    try:
        apps = paginator.page(page)
    except PageNotAnInteger:
        # If page is not an integer, deliver first page.
        apps = paginator.page(1)
    except EmptyPage:
        # If page is out of range (e.g. 9999), deliver last page of results.
        apps = paginator.page(paginator.num_pages)
    context = {"my_appointments": apps}
    return render(request, 'pat/my_apps.html', context)

@login_required(login_url='/')
def CancelAppointment(request, id):
    apmt = get_object_or_404(Appointment, id=id)
    if apmt:
        apmt.delete()
        messages.success(request, "Appointment Cancelled Successfully.")
        return redirect('my_apps')

@login_required(login_url='/')
def MedicalHistory(request, id):
    user = get_object_or_404(CustomUser, id=id)
    patient = get_object_or_404(PatientReg, admin=user)
    all_appointments = Appointment.objects.filter(patient=patient, status='Completed').order_by('-date_of_appointment')
    # Pagination
    paginator = Paginator(all_appointments, 5)  # Show 10 appointments per page
    page = request.GET.get('page')
    try:
        all_appointments = paginator.page(page)
    except PageNotAnInteger:
        # If page is not an integer, deliver first page.
        all_appointments = paginator.page(1)
    except EmptyPage:
        # If page is out of range (e.g. 9999), deliver last page of results.
        all_appointments = paginator.page(paginator.num_pages)
    context = {
        'appointments': all_appointments
    }
    return render(request, 'pat/med_history.html', context)