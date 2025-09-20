from django.db import models
from django.contrib.auth.models import AbstractUser

class CustomUser(AbstractUser):
    USER ={
        (1,'admin'),
        (2,'doc'),
        (3, 'patient')
        
    }
    user_type = models.CharField(choices=USER,max_length=50,default=1)

    profile_pic = models.ImageField(upload_to='media/profile_pic')

class Specialization(models.Model):
    sname = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.sname
   
    

class DoctorReg(models.Model):
    admin = models.OneToOneField(CustomUser, on_delete=models.CASCADE, null=True, blank=True)
   
    mobilenumber = models.CharField(max_length=11)
    specialization_id = models.ForeignKey(Specialization, on_delete=models.CASCADE, null=True, blank=True)
    consultation_start = models.TimeField(null=True, blank=True)
    consultation_end = models.TimeField(null=True, blank=True)
    dalily_patients = models.IntegerField(null=True, blank=True)
    consultation_fee = models.IntegerField(default=200)
    regdate_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        if self.admin:
            return f"{self.admin.first_name} {self.admin.last_name} - {self.specialization_id}"
        else:
            return f"User not associated - {self.mobilenumber}"

class PatientReg(models.Model):
    admin = models.OneToOneField(CustomUser, on_delete=models.CASCADE, null=True, blank=True)
    mobilenumber = models.CharField(max_length=11)
    email = models.EmailField()
    regdate_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        if self.admin:
            return f"{self.admin.first_name} {self.admin.last_name} - {self.mobilenumber}"
        else:
            return f"User not associated - {self.mobilenumber}"
   

class Appointment(models.Model):
    patient = models.ForeignKey(PatientReg, on_delete=models.CASCADE, null=True, blank=True)
    appointmentnumber = models.IntegerField(default=0)
    fullname = models.CharField(max_length=250)
    mobilenumber = models.CharField(max_length=11)
    email = models.EmailField(max_length=100)
    date_of_appointment = models.CharField(max_length=250)
    time_of_appointment = models.CharField(max_length=250, null=True, blank=True)
    token = models.IntegerField(null=True, blank=True)
    doctor_id = models.ForeignKey(DoctorReg, on_delete=models.CASCADE)
    additional_msg = models.TextField(blank=True)
    remark = models.CharField(max_length=250,default=0)
    status = models.CharField(default=0,max_length=200)
    prescription=models.TextField(blank=True,default=0)
    findings=models.TextField(blank=True,default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Appointment #{self.appointmentnumber} - {self.fullname}"

class Page(models.Model):
    pagetitle = models.CharField(max_length=250)
    address = models.CharField(max_length=250)
    aboutus = models.TextField()
    email = models.EmailField(max_length=200)
    mobilenumber = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.pagetitle
    
class Payments(models.Model):
    appointment = models.ForeignKey(Appointment, on_delete=models.DO_NOTHING)
    card_name = models.CharField(max_length=100)
    card_no = models.CharField(max_length=50)
    expiry = models.CharField(max_length=10)
    cvv = models.CharField(max_length=5)
    pincode = models.CharField(max_length=10)
    amount = models.FloatField()
    admin_share = models.FloatField()
    doctor_share = models.FloatField()
    create_date = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return str(self.appointment)
 
