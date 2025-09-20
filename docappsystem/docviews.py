from django.shortcuts import get_object_or_404, render,redirect,HttpResponse
from django.contrib.auth.decorators import login_required
from dasapp.models import DoctorReg,Specialization,CustomUser,Appointment, Payments
from django.contrib import messages
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from datetime import datetime, timedelta
from .userviews import *

def DOCSIGNUP(request):
    specialization = Specialization.objects.all()
    if request.method == "POST":
        pic = request.FILES.get('pic')
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        username = request.POST.get('username')
        email = request.POST.get('email')
        mobno = request.POST.get('mobno')
        specialization_id = request.POST.get('specialization_id')
        starttime = request.POST.get('starttime')
        endtime = request.POST.get('endtime')
        limit = request.POST.get('patientscount')
        consultation_fee = request.POST.get('consult_fee')
        password = request.POST.get('password')
        
        try:
            starttime24 = datetime.strptime(starttime, "%I:%M %p").time()
            endtime24 = datetime.strptime(endtime,"%I:%M %p").time()
        except:
            messages.warning(request,'Please provide time in the specified format.')
            return redirect('docsignup')

        if CustomUser.objects.filter(email=email).exists():
            messages.warning(request,'Email already exist')
            return redirect('docsignup')
        if CustomUser.objects.filter(username=username).exists():
            messages.warning(request,'Username already exist')
            return redirect('docsignup')
        else:
            user = CustomUser(
               first_name=first_name,
               last_name=last_name,
               username=username,
               email=email,
               user_type=2,
               profile_pic = pic,
            )
            user.set_password(password)
            user.save()
            spid =Specialization.objects.get(id=specialization_id)
            doctor = DoctorReg(
                admin = user,
                
                mobilenumber = mobno,
                specialization_id = spid,
                consultation_start = starttime24,
                consultation_end = endtime24,
                dalily_patients = limit,
                consultation_fee = consultation_fee
                
            )
            doctor.save()            
            messages.success(request,'Signup Successfully')
            return redirect('docsignup')
    
    context = {
        'specialization':specialization
    }

    return render(request,'doc/docreg.html',context)

@login_required(login_url='/')
def DOCTORHOME(request):
    doctor_admin = request.user
    doctor_reg = DoctorReg.objects.get(admin=doctor_admin)
    allaptcount = Appointment.objects.filter(doctor_id=doctor_reg).count
    newaptcount = Appointment.objects.filter(status='0',doctor_id=doctor_reg).count
    appaptcount = Appointment.objects.filter(status='Approved',doctor_id=doctor_reg).count
    canaptcount = Appointment.objects.filter(status='Cancelled',doctor_id=doctor_reg).count
    comaptcount = Appointment.objects.filter(status='Completed',doctor_id=doctor_reg).count
    tot_payments = Payments.objects.filter(appointment__doctor_id = doctor_reg)
    revenue = 0
    for payment in tot_payments:
        revenue += payment.doctor_share
    context = {
        'newaptcount':newaptcount,
        'allaptcount':allaptcount,
        'appaptcount':appaptcount,
        'canaptcount':canaptcount,
        'comaptcount':comaptcount,
        'revenue': revenue
    }
    return render(request,'doc/dochome.html',context)

@login_required(login_url='/')
def View_Appointment(request):
    try:
        doctor_admin = request.user
        doctor_reg = DoctorReg.objects.get(admin=doctor_admin)
        view_appointment = Appointment.objects.filter(doctor_id=doctor_reg).order_by('-pk')
        

        # Pagination
        paginator = Paginator(view_appointment, 5)  # Show 10 appointments per page
        page = request.GET.get('page')
        try:
            view_appointment = paginator.page(page)
        except PageNotAnInteger:
            # If page is not an integer, deliver first page.
            view_appointment = paginator.page(1)
        except EmptyPage:
            # If page is out of range (e.g. 9999), deliver last page of results.
            view_appointment = paginator.page(paginator.num_pages)

        context = {'view_appointment': view_appointment}
    except Exception as e:
        # Handle exceptions, such as database errors, gracefully
        context = {'error_message': str(e)}

    return render(request, 'doc/view_appointment.html', context)

@login_required(login_url='/')
def Patient_Appointment_Details(request,id):
    patientdetails=Appointment.objects.filter(id=id).order_by('-pk')
    context={'patientdetails':patientdetails

    }

    return render(request,'doc/patient_appointment_details.html',context)

def calculateTimeDifference(starttime, endtime):
    start_datetime = datetime.combine(datetime.today(), starttime)
    end_datetime = datetime.combine(datetime.today(), endtime)
    timediff = end_datetime-start_datetime
    if timediff.days < 0:
        end_datetime = datetime.combine(datetime.today(), endtime)+timedelta(days=1)
        timediff = end_datetime - start_datetime
    minutesDiff = timediff.total_seconds()/60
    return minutesDiff

def addMinutes(time, minutes):
    time_datetime = datetime.combine(datetime.today(), time)
    new_datetime = time_datetime + timedelta(minutes=minutes)
    new_time = new_datetime.time()
    return new_time

@login_required(login_url='/')
def Patient_Allot_Time(request, app_id):
    if request.method == 'POST':
        card_name = request.POST.get('card_name')
        card_no = request.POST.get('card_no')
        expiry = request.POST.get('expiry')
        cvv = request.POST.get('cvv')
        pincode = request.POST.get('pincode')
        
        print(card_name, card_no, expiry, cvv, pincode)
        
        patientaptdet = Appointment.objects.get(id=app_id)
        
        amount = patientaptdet.doctor_id.consultation_fee
        admin_share = (amount * 0.10)
        doctor_share = amount - admin_share
        
        payment_obj = Payments.objects.create(
            appointment = patientaptdet,
            card_name = card_name,
            card_no = card_no,
            expiry = expiry,
            cvv = cvv,
            pincode = pincode,
            amount = amount,
            admin_share = admin_share,
            doctor_share = doctor_share
        )
        if payment_obj:
            doctor = patientaptdet.doctor_id
            date = patientaptdet.date_of_appointment
            aptcount = Appointment.objects.filter(date_of_appointment=date, doctor_id=doctor, status='PaymentDone').count()
            timediff = calculateTimeDifference(doctor.consultation_start, doctor.consultation_end)
            minsPerPerson = timediff/doctor.dalily_patients
            minsToAdd = minsPerPerson*(aptcount)
            if minsToAdd > 0:
                apt_time = addMinutes(doctor.consultation_start, minsToAdd)
            else:
                apt_time = doctor.consultation_start
            apt_time12 = apt_time.strftime("%I:%M %p")
            patientaptdet.time_of_appointment = apt_time12
            patientaptdet.token = int(aptcount)+1
            patientaptdet.status = 'PaymentDone'
            patientaptdet.save()
            messages.success(request,"Payment Done and Token & Time alloted for you.")
            return redirect('viewappointmentdetails', app_id)
        else:
            messages.error(request,"Payment cannot be proceeded.")
            return redirect('user_search_appointment')

@login_required(login_url='/')
def Patient_Appointment_Details_Remark(request):
    if request.method == 'POST':
        patient_id = request.POST.get('pat_id')
        remark = request.POST['remark']
        status = request.POST['status']
        patientaptdet = Appointment.objects.get(id=patient_id)
        patientaptdet.remark = remark
        if status == 'Approved':
            patient = get_object_or_404(PatientReg, id=patientaptdet.patient.id)
            doc_inst = DoctorReg.objects.get(admin=request.user)
            doa = datetime.strptime(patientaptdet.date_of_appointment, "%Y-%m-%d").date()
            fifteen_days_ago = doa - timedelta(days=15)
            previous_appointment = Appointment.objects.filter(
                patient=patient,
                doctor_id=doc_inst,
                date_of_appointment__gte=fifteen_days_ago,
                date_of_appointment__lt=patientaptdet.date_of_appointment,
                status__in=["PaymentDone", "Completed"]
            ).first()
            if previous_appointment:
                date = patientaptdet.date_of_appointment
                aptcount = Appointment.objects.filter(date_of_appointment=date, doctor_id=doc_inst, status='PaymentDone').count()
                timediff = calculateTimeDifference(doc_inst.consultation_start, doc_inst.consultation_end)
                minsPerPerson = timediff/doc_inst.dalily_patients
                minsToAdd = minsPerPerson*(aptcount)
                if minsToAdd > 0:
                    apt_time = addMinutes(doc_inst.consultation_start, minsToAdd)
                else:
                    apt_time = doc_inst.consultation_start
                apt_time12 = apt_time.strftime("%I:%M %p")
                patientaptdet.time_of_appointment = apt_time12
                patientaptdet.token = int(aptcount)+1
                patientaptdet.status = 'PaymentDone'
            else:
                patientaptdet.status = status
        else:
            patientaptdet.status = status
        patientaptdet.save()
            
        messages.success(request,"Status Update successfully")
        return redirect('view_appointment')
    return render(request,'doc/view_appointment.html',context)

@login_required(login_url='/')
def Patient_Approved_Appointment(request):
    doctor_admin = request.user
    doctor_reg = DoctorReg.objects.get(admin=doctor_admin)
    patientdetails1 = Appointment.objects.filter(status='PaymentDone',doctor_id=doctor_reg).order_by('-pk')
    context = {'patientdetails1': patientdetails1}
    return render(request, 'doc/patient_app_appointment.html', context)

@login_required(login_url='/')
def Patient_Cancelled_Appointment(request):
    doctor_admin = request.user
    doctor_reg = DoctorReg.objects.get(admin=doctor_admin)
    patientdetails1 = Appointment.objects.filter(status='Cancelled',doctor_id=doctor_reg).order_by('-pk')
    context = {'patientdetails1': patientdetails1}
    return render(request, 'doc/patient_app_appointment.html', context)

@login_required(login_url='/')
def Patient_New_Appointment(request):
    doctor_admin = request.user
    doctor_reg = DoctorReg.objects.get(admin=doctor_admin)
    patientdetails1 = Appointment.objects.filter(status='0',doctor_id=doctor_reg).order_by('-pk')
    context = {'patientdetails1': patientdetails1}
    return render(request, 'doc/patient_app_appointment.html', context)

@login_required(login_url='/')
def Patient_List_Approved_Appointment(request):
    doctor_admin = request.user
    doctor_reg = DoctorReg.objects.get(admin=doctor_admin)
    patientdetails1 = Appointment.objects.filter(status='PaymentDone',doctor_id=doctor_reg).order_by('-pk')
    context = {'patientdetails1': patientdetails1}
    return render(request, 'doc/patient_list_app_appointment.html', context)

@login_required(login_url='/')
def DoctorAppointmentList(request, id):
    patientdetails = Appointment.objects.filter(id=id)
    appt = get_object_or_404(Appointment, id=id)
    patient = appt.patient
    all_appointments = Appointment.objects.filter(patient=patient, status='Completed').order_by('-date_of_appointment').order_by('-pk')
    context = {
        'patientdetails': patientdetails,
        'appointments': all_appointments
    }

    return render(request,'doc/doctor_appointment_list_details.html',context)

@login_required(login_url='/')
def Patient_Appointment_Prescription(request):
    if request.method == 'POST':
        patient_id = request.POST.get('pat_id')
        prescription = request.POST['prescription']
        findings = request.POST['findings']
        status = request.POST['status']
        patientaptdet = Appointment.objects.get(id=patient_id)
        patientaptdet.prescription = prescription
        patientaptdet.findings = findings
        patientaptdet.status = status
        patientaptdet.save()
        messages.success(request,"Status Update successfully")
        return redirect('view_appointment')
    return render(request,'doc/patient_list_app_appointment.html',context)

@login_required(login_url='/')
def Patient_Appointment_Completed(request):
    doctor_admin = request.user
    doctor_reg = DoctorReg.objects.get(admin=doctor_admin)
    patientdetails1 = Appointment.objects.filter(status='Completed',doctor_id=doctor_reg).order_by('-pk')
    context = {'patientdetails1': patientdetails1}
    return render(request, 'doc/patient_list_app_appointment.html', context)

@login_required(login_url='/')
def Search_Appointments(request):
    doctor_admin = request.user
    doctor_reg = DoctorReg.objects.get(admin=doctor_admin.id)
    if request.method == "GET":
        query = request.GET.get('query', '')
        if query:
            # Filter records where fullname or Appointment Number contains the query
            patient = Appointment.objects.filter(fullname__icontains=query) | Appointment.objects.filter(appointmentnumber__icontains=query) & Appointment.objects.filter(doctor_id=doctor_reg)
            messages.success(request, "Search against " + query)
            return render(request, 'doc/search-appointment.html', {'patient': patient, 'query': query})
        else:
            print("No Record Found")
            return render(request, 'doc/search-appointment.html', {})

@login_required(login_url='/')
def Between_Date_Report(request):
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')

    patient = []
    doctor_admin = request.user
    doctor_reg = DoctorReg.objects.get(admin=doctor_admin)

    if start_date and end_date:
        # Validate the date inputs
        try:
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        except ValueError:
            return render(request, 'doc/between-dates-report.html', {'visitor': visitor, 'error_message': 'Invalid date format'})

        # Filter Appointment between the given date range
        patient = Appointment.objects.filter(created_at__range=(start_date, end_date)) & Appointment.objects.filter(doctor_id=doctor_reg)

    return render(request, 'doc/between-dates-report.html', {'patient': patient,'start_date':start_date,'end_date':end_date})
