# Generated by Django 4.2.5 on 2024-01-01 13:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('simulation', '0008_alter_vote_timestamp_alter_votebackup_timestamp'),
    ]

    operations = [
        migrations.AddField(
            model_name='vote',
            name='status',
            field=models.CharField(choices=[('valid', 'Valid'), ('rejected', 'Rejected')], default='valid', max_length=8),
        ),
    ]
