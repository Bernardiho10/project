# Generated by Django 4.2.5 on 2024-01-01 13:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('simulation', '0009_vote_status'),
    ]

    operations = [
        migrations.AddField(
            model_name='votebackup',
            name='status',
            field=models.CharField(choices=[('valid', 'Valid'), ('rejected', 'Rejected')], default='valid', max_length=8),
        ),
    ]
