from pyexpat import model
from django.db import models

# Create your models here.

class testing(models.Model):
    symbol = models.CharField(max_length=12)
    date = models.CharField(max_length=15)


class files_format(models.Model):
    csv_file = models.FileField(db_column='csv_file', max_length=255, blank=True)

    class Meta:
        db_table = 'main'