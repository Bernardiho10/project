import datetime, uuid, random
from django.db import models

def get_vote():
    return random.randint(1, 3)

def get_timestamp():
    return datetime.datetime.now().timestamp()

class Vote(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    vote = models.CharField(max_length=5)
    timestamp = models.CharField(max_length=100)
    ip_address = models.GenericIPAddressField(null=True)
    mac_address = models.CharField(max_length=17, null=True)
    nin = models.CharField(max_length=255, null=True)
    inec = models.CharField(max_length=255, null=True)
    state = models.CharField(max_length=30, null=True)
    # Not ForeignKey! See transactions() in simulation.views for implications
    block_id = models.IntegerField(null=True)

    def __str__(self):
        return "{}|{}|{}".format(self.id, self.vote,self.state, self.nin, self.inec, self.ip_address, self.mac_address, self.timestamp)

class Block(models.Model):
    prev_h = models.CharField(max_length=64, blank=True)
    merkle_h = models.CharField(max_length=64, blank=True)
    h = models.CharField(max_length=64, blank=True)
    nonce = models.IntegerField(null=True)
    timestamp = models.CharField(max_length=100)

    def __str__(self):
        return str(self.id)

class VoteBackup(models.Model):
    """This model acts as backup; its objects shall never be tampered."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    vote = models.CharField(max_length=5)
    timestamp = models.CharField(max_length=100)
    ip_address = models.GenericIPAddressField(null=True)
    mac_address = models.CharField(max_length=17, null=True)
    nin = models.CharField(max_length=255, null=True)
    state = models.CharField(max_length=30, null=True)
    inec = models.CharField(max_length=255, null=True)
    block_id = models.IntegerField(null=True)

    def __str__(self):
        return "{}|{}|{}".format(self.id, self.vote, self.state,  self.nin, self.inec, self.ip_address, self.mac_address,  self.timestamp)
