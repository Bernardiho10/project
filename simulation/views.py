import datetime, time, json, math
import os
import random
import pandas as pd
import random
from faker import Faker
from random import randint
from uuid import uuid4
from Crypto.Hash import SHA3_256
from django.core.paginator import EmptyPage, PageNotAnInteger, Paginator
from django.conf import settings
from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, render, redirect
from datetime import timedelta
from .models import Vote, Block, VoteBackup
from .merkle.merkle_tool import MerkleTools
import csv


def generate_users_from_excel(request):
    # Read the historical voting data from the Excel file
    excel_file = os.path.join(os.path.dirname(__file__),
                              'kaggle-DataCleaned-2015elections.xls')  # Provide the path to your Excel file
    df = pd.read_excel(excel_file)

    # Shuffle DataFrame Rows
    df = df.sample(frac=1).reset_index(drop=True)

    # Delete all data from the previous demo.
    deleted_old_votes = Vote.objects.all().delete()[0]
    VoteBackup.objects.all().delete()
    print(f'\nDeleted {deleted_old_votes} data from the previous simulation.\n')

    # Initialize counters for APC and PDP votes
    apc_votes_total = 0
    pdp_votes_total = 0

    # Generate users based on historical data
    time_start = time.time()
    block_no = 1
    current_time = time.time()

    for index, row in df.iterrows():
        state = row['State']
        apc_votes = row['APC Votes']
        pdp_votes = row['PDP Votes']
        total_votes = row['Votes cast']
        valid_votes = row['Valid votes']
        rejected_votes = row['Rejected votes']

        # Determine the number of valid and rejected votes for APC and PDP
        valid_apc_votes = int((apc_votes / total_votes) * valid_votes)
        valid_pdp_votes = int((pdp_votes / total_votes) * valid_votes)
        rejected_apc_votes = apc_votes - valid_apc_votes
        rejected_pdp_votes = pdp_votes - valid_pdp_votes

        # Create a list of votes with status
        votes = [('APC', 'valid')] * valid_apc_votes + [('PDP', 'valid')] * valid_pdp_votes + \
                [('APC', 'rejected')] * rejected_apc_votes + [('PDP', 'rejected')] * rejected_pdp_votes
        random.shuffle(votes)

        for vote, status in votes:
            # Introduce some randomness in timestamps (realistic time window)
            timestamp = current_time - random.uniform(0, 60 * 60 * 3)
            formatted_time = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

            generate_user(state, 1 if vote == 'APC' else 2, block_no, formatted_time, status)

            if vote == 'APC':
                apc_votes_total += 1
            else:
                pdp_votes_total += 1

            #  Introduce a delay between generating each user
            time.sleep(0.3)

        block_no += 1

    time_end = time.time()
    print(f'\nFinished in {(time_end - time_start)} seconds.\n')
    print(f'Total APC Votes Created: {apc_votes_total}')
    print(f'Total PDP Votes Created: {pdp_votes_total}')

    # View the generated transactions
    votes = Vote.objects.order_by('-timestamp')[:100]  # Only shows the last 100, if any
    context = {
        'votes': votes,
    }
    request.session['transactions_done'] = True
    return render(request, 'simulation/generate.html', context)




def generate_user(state, vote, block_no, timestamp, vote_status):
    if timestamp is None:
        timestamp = round(_get_timestamp())

    v_id = str(uuid4())
    v_ip = _get_ipaddress()


    # Read the MAC address Excel file
    excel_file = 'nigeria_mac_addy.xls'
    df = pd.read_excel(excel_file)

    # Normalize the state names in the column headers
    df.columns = df.columns.str.strip().str.lower()

    # Check if the state exists in the DataFrame columns
    if state.lower() in df.columns:
        # Select the column for the state and drop any NaN values
        state_column = df[state.lower()].dropna()

        # Randomly select a MAC address if the column is not empty
        v_mac = random.choice(state_column.tolist()) if not state_column.empty else None
    else:
        # Handle invalid state
        print("Error: State Not found!")
        v_mac = None


    v_nin = _get_nin()
    v_inec = _get_inec()

    new_vote = Vote(
        id=v_id,
        vote=vote,
        state=state,
        nin=v_nin,
        inec=v_inec,
        ip_address=v_ip,
        mac_address=v_mac,
        status=vote_status,
        timestamp=timestamp,
        block_id=block_no,
    )
    new_backup_vote = VoteBackup(
        id=v_id,
        vote=vote,
        state=state,
        nin=v_nin,
        inec=v_inec,
        ip_address=v_ip,
        mac_address=v_mac,
        status=vote_status,
        timestamp=timestamp,
        block_id=block_no,
    )

    # "Broadcast" to two nodes
    new_vote.save()
    new_backup_vote.save()
    print(f"Generated user for {state} - Vote: {vote}, Block: {block_no}, Timestamp: {timestamp}")

    return new_vote


def seal(request):
    """Seal the transactions generated previously."""
    if request.session.get('transactions_done') is None:
        redirect('welcome:home')
    del request.session['transactions_done']

    # Puzzle requirement: '0' * (n leading zeros)
    puzzle, pcount = settings.PUZZLE, settings.PLENGTH

    # Seal transactions into blocks
    time_start = time.time()
    number_of_blocks = settings.N_BLOCKS
    prev_hash = '0' * 64
    for i in range(1, number_of_blocks + 1):
        block_transactions = Vote.objects.filter(block_id=i).order_by('timestamp')
        root = MerkleTools()
        root.add_leaf([str(tx) for tx in block_transactions], True)
        root.make_tree()
        merkle_h = root.get_merkle_root()

        # Try to seal the block and generate valid hash
        nonce = 0
        timestamp = datetime.datetime.now().timestamp()
        while True:
            enc = ("{}{}{}{}".format(prev_hash, merkle_h, nonce, timestamp)).encode('utf-8')
            h = SHA3_256.new(enc).hexdigest()
            if h[:pcount] == puzzle:
                break
            nonce += 1

        # Create the block
        block = Block(id=i, prev_h=prev_hash, merkle_h=merkle_h, h=h, nonce=nonce, timestamp=timestamp)
        block.save()
        print('\nBlock {} is mined\n'.format(i))
        # Set this hash as prev hash
        prev_hash = h

    time_end = time.time()
    print('\nSuccessfully created {} blocks.\n'.format(number_of_blocks))
    print('\nFinished in {} seconds.\n'.format(time_end - time_start))
    return redirect('simulation:blockchain')

def transactions(request):
    """See all transactions that have been contained in blocks."""
    vote_list = Vote.objects.all().order_by('timestamp')
    paginator = Paginator(vote_list, 100, orphans=20, allow_empty_first_page=True)

    page = request.GET.get('page')
    votes = paginator.get_page(page)

    hashes = [SHA3_256.new(str(v).encode('utf-8')).hexdigest() for v in votes]

    # This happens if you don't use foreign key
    block_hashes = []
    for i in range(0, len(votes)):
        try:
            b = Block.objects.get(id=votes[i].block_id)
            h = b.h
        except:
            h = 404
        block_hashes.append(h)

    # zip the three iters
    votes_pg = votes # for pagination
    votes = zip(votes, hashes, block_hashes)

    # Calculate the voting result of 3 cands, the ugly way
    result = []
    for i in range(0, 3):
        try:
            r = Vote.objects.filter(vote=i+1).count()
        except:
            r = 0
        result.append(r)

    context = {
        'votes': votes,
        'result': result,
        'votes_pg': votes_pg,
    }
    return render(request, 'simulation/transactions.html', context)

def blockchain(request):
    """See all mined blocks."""
    blocks = Block.objects.all().order_by('id')
    context = {
        'blocks': blocks,
    }
    return render(request, 'simulation/blockchain.html', context)

def verify(request):
    """Verify transactions in all blocks by re-calculating the merkle root."""
    # Basically, by just creating a session (message) var

    print('verifying data...')
    number_of_blocks = Block.objects.all().count()
    corrupt_block_list = ''
    for i in range(1, number_of_blocks + 1):
        # Select block #i
        b = Block.objects.get(id=i)

        # Select all transactions in block #i
        transactions = Vote.objects.filter(block_id=i).order_by('timestamp')

        # Verify them
        root = MerkleTools()
        root.add_leaf([str(tx) for tx in transactions], True)
        root.make_tree()
        merkle_h = root.get_merkle_root()

        if b.merkle_h == merkle_h:
            message = 'Block {} verified.'.format(i)
        else:
            message = 'Block {} is TAMPERED'.format(i)
            corrupt_block_list += ' {}'.format(i)
        print('{}'.format(message))
    if len(corrupt_block_list) > 0:
        messages.warning(request, 'The following blocks have corrupted transactions: {}.'.format(corrupt_block_list), extra_tags='bg-danger')
    else:
        messages.info(request, 'All transactions in blocks are intact.', extra_tags='bg-info')
    return redirect('simulation:blockchain')

def sync(request):
    """Restore transactions from honest node."""
    deleted_old_votes = Vote.objects.all().delete()[0]
    print('\nTrying to sync {} transactions with 1 node(s)...\n'.format(deleted_old_votes))
    bk_votes = VoteBackup.objects.all().order_by('timestamp')
    for bk_v in bk_votes:
        vote = Vote(id=bk_v.id, state=bk_v.state, vote=bk_v.vote, nin=bk_v.nin, inec=bk_v.inec, ip_address=bk_v.ip_address,  mac_address=bk_v.mac_address, timestamp=bk_v.timestamp, block_id=bk_v.block_id)
        vote.save()
    print('\nSync complete.\n')
    messages.info(request, 'All blocks have been synced successfully.')
    return redirect('simulation:blockchain')

def sync_block(request, block_id):
    """Restore transactions of a block from honest node."""
    b = Block.objects.get(id=block_id)
    print('\nSyncing transactions in block {}\n'.format(b.id))
    # Get all existing transactions in this block and delete them
    Vote.objects.filter(block_id=block_id).delete()
    # Then rewrite from backup node
    bak_votes = VoteBackup.objects.filter(block_id=block_id).order_by('timestamp')
    for bv in bak_votes:
        v = Vote(id=bv.id, state=bv.state, vote=bv.vote, nin=bv.nin, inec=bv.inec, ip_address=bv.ip_address, geographic_region=bv.geographic_region, mac_address=bv.mac_address, timestamp=bv.timestamp, block_id=bv.block_id)
        v.save()
    # Just in case, delete transactions without valid block
    block_count = Block.objects.all().count()
    Vote.objects.filter(block_id__gt=block_count).delete()
    Vote.objects.filter(block_id__lt=1).delete()
    print('\nSync complete\n')
    return redirect('simulation:block_detail', block_hash=b.h)

def block_detail(request, block_hash):
    """See the details of a block and its transactions."""
    # Select the block or 404
    block = get_object_or_404(Block, h=block_hash)
    confirmed_by = (Block.objects.all().count() - block.id) + 1
    # Select all corresponding transactions
    transaction_list = Vote.objects.filter(block_id=block.id).order_by('timestamp')
    paginator = Paginator(transaction_list, 100, orphans=20)

    page = request.GET.get('page')
    transactions = paginator.get_page(page)
    transactions_hashes = [SHA3_256.new(str(t).encode('utf-8')).hexdigest() for t in transactions]

    # Check the integrity of transactions
    root = MerkleTools()
    root.add_leaf([str(tx) for tx in transaction_list], True)
    root.make_tree()
    merkle_h = root.get_merkle_root()
    tampered = block.merkle_h != merkle_h

    transactions_pg = transactions # for pagination
    transactions = zip(transactions, transactions_hashes)

    # Get prev and next block id
    prev_block = Block.objects.filter(id=block.id - 1).first()
    next_block = Block.objects.filter(id=block.id + 1).first()

    context = {
        'bk': block,
        'confirmed_by': confirmed_by,
        'transactions': transactions,
        'tampered': tampered,
        'verified_merkle_h': merkle_h,
        'prev_block': prev_block,
        'next_block': next_block,
        'transactions_pg': transactions_pg,
    }

    return render(request, 'simulation/block.html', context)

nigerian_geographical_zones = {
    'Abia': 'South East',
    'Adamawa': 'North East',
    'Akwa Ibom': 'South South',
    'Anambra': 'South East',
    'Bauchi': 'North East',
    'Bayelsa': 'South South',
    'Benue': 'North Central',
    'Borno': 'North East',
    'Cross River': 'South South',
    'Delta': 'South South',
    'Ebonyi': 'South East',
    'Edo': 'South South',
    'Ekiti': 'South West',
    'Enugu': 'South East',
    'Gombe': 'North East',
    'Imo': 'South East',
    'Jigawa': 'North West',
    'Kaduna': 'North West',
    'Kano': 'North West',
    'Katsina': 'North West',
    'Kebbi': 'North West',
    'Kogi': 'North Central',
    'Kwara': 'North Central',
    'Lagos': 'South West',
    'Nasarawa': 'North Central',
    'Niger': 'North Central',
    'Ogun': 'South West',
    'Ondo': 'South West',
    'Osun': 'South West',
    'Oyo': 'South West',
    'Plateau': 'North Central',
    'Rivers': 'South South',
    'Sokoto': 'North West',
    'Taraba': 'North East',
    'Yobe': 'North East',
    'Zamfara': 'North West',
    'FCT': 'North Central'
}

def export_transactions_to_csv(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="transactions.csv"'

    writer = csv.writer(response)

    # Read Excel file and convert relevant columns to integers
    excel_file = os.path.join(os.path.dirname(__file__),
                              'kaggle-DataCleaned-2015elections.xls')
    df = pd.read_excel(excel_file)
    df['Accredited Voters'] = df['Accredited Voters'].astype(int)
    df['Votes cast'] = df['Votes cast'].astype(int)
    df['Valid votes'] = df['Valid votes'].astype(int)
    df['Rejected votes'] = df['Rejected votes'].astype(int)

    # Calculate global sums
    accredited_voters_sum = df['Accredited Voters'].sum()
    votes_cast_sum = df['Votes cast'].sum()
    valid_votes_sum = df['Valid votes'].sum()
    rejected_votes_sum = df['Rejected votes'].sum()

    # Write headers for transactions and sums
    writer.writerow(['Transaction ID', 'Vote', 'State', 'NIN', 'Ip Address',
                     'INEC', 'Center Mac Address', 'Timestamp', 'Block ID',
                     'Vote Status', 'Gender', 'Occupation', 'Age', 'Geographical Zones', 'Accredited Voters', 'Votes Cast', 'Valid Votes',
                     'Rejected Votes'])

    accredited_voters = df['Accredited Voters'].sum()
    female_percentage = 0.53
    male_percentage = 1 - female_percentage

    occupations = {
        'Student': 0.2657,
        'Farming and fishing': 0.1623,
        'Housewives': 0.141,
        'Business sector': 0.1287,
        'Traders': 0.0901,
        'Uncategorised': 0.0717,
        'Civil servants': 0.06,
        'Artisans': 0.0533,
        'Public servants': 0.0273
    }

    age_groups = {
        '18-35': 0.511,
        '36-50': 0.299,
        '51+': 0.19
    }


    transactions = Vote.objects.all()

    for transaction in transactions:
        gender = 'Female' if random.random() < female_percentage else 'Male'

        occupation = random.choices(list(occupations.keys()), weights=list(occupations.values()))[0]

        if random.random() < 0.511:
            age_group = random.randint(18, 35)
        elif random.random() < 0.811:
            age_group = random.randint(36, 50)
        else:
            age_group = random.randint(51, 100)

        geographical_zone = nigerian_geographical_zones.get(transaction.state, 'Unknown')

        writer.writerow([transaction.id, transaction.vote, transaction.state,
                         transaction.nin, transaction.ip_address, transaction.inec,
                         transaction.mac_address, transaction.timestamp,
                         transaction.block_id, transaction.status, gender, occupation, age_group,
                         geographical_zone,  # New column data
                         '', '', '', ''])

    # Write first row with sums
    writer.writerow(['', '', '', '', '', '', '', '', '',
                         '', '', '', '', '', accredited_voters_sum, votes_cast_sum,
                         valid_votes_sum, rejected_votes_sum])

    return response





# HELPER FUNCTIONS
def _get_vote():
    return randint(1, 3)

def _get_timestamp():
    return datetime.datetime.now().timestamp()


def _get_ipaddress():
    random_ipv4 = ".".join(str(random.randint(0, 255)) for _ in range(4))
    return random_ipv4


def _get_mac_address():
    return ':'.join(['{:02x}'.format(random.randint(0, 255)) for _ in range(6)])


def _get_nin():
    return randint(10**8, 10**9 - 1)

def _get_inec():
    return randint(10**8, 10**9 - 1)