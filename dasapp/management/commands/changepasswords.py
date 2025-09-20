from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from dasapp.models import CustomUser
from django.contrib.auth.hashers import make_password

class Command(BaseCommand):
    help = 'Change the password for all non-superusers'

    def add_arguments(self, parser):
        parser.add_argument('new_password', type=str, help='The new password to set for all non-superusers')

    def handle(self, *args, **kwargs):
        new_password = kwargs['new_password']
        hashed_password = make_password(new_password)
        
        # Update passwords only for non-superusers
        users_to_update = CustomUser.objects.filter(is_superuser=False)
        users_to_update.update(password=hashed_password)

        self.stdout.write(self.style.SUCCESS(f'Successfully updated passwords for {users_to_update.count()} non-superuser(s)'))
