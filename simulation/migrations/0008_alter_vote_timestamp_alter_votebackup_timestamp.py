# Generated by Django 4.2.5 on 2023-12-17 22:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('simulation', '0007_alter_vote_timestamp_alter_votebackup_timestamp'),
    ]

    operations = [
        migrations.AlterField(
            model_name='vote',
            name='timestamp',
            field=models.CharField(max_length=100),
        ),
        migrations.AlterField(
            model_name='votebackup',
            name='timestamp',
            field=models.CharField(max_length=100),
        ),
    ]
