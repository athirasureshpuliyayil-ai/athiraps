from django.core.management.base import BaseCommand
from dasapp.models import CustomUser

class Command(BaseCommand):
    help = 'Fetch all email addresses from the User table'

    def handle(self, *args, **kwargs):
        emails = CustomUser.objects.values_list('email', flat=True)
        for email in emails:
            self.stdout.write(email)
