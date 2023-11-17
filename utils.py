# utils.py

import os
import shutil
import requests
import json
from zipfile import ZipFile
import hashlib
import base64
import datetime
import random
import string
import re
import urllib.parse


def download_file(url, destination):
    # Function to download a file from a URL
    response = requests.get(url)
    with open(destination, 'wb') as f:
        f.write(response.content)


def extract_zip(file_path, extract_to):
    # Function to extract a zip file
    with ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def hash_file(file_path):
    # Function to calculate hash of a file
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def encode_base64(data):
    # Function to encode data to base64
    return base64.b64encode(data.encode()).decode()


def decode_base64(data):
    # Function to decode base64 data
    return base64.b64decode(data.encode()).decode()


def generate_random_string(length):
    # Function to generate a random string
    letters_and_digits = string.ascii_letters + string.digits
    return ''.join(random.choice(letters_and_digits) for i in range(length))


def get_current_timestamp():
    # Function to get current timestamp
    return datetime.datetime.now().timestamp()


def remove_special_characters(text):
    # Function to remove special characters from text
    return re.sub(r'[^A-Za-z0-9 ]+', '', text)


def parse_query_params(url):
    # Function to parse query parameters from a URL
    parsed = urllib.parse.urlparse(url)
    return urllib.parse.parse_qs(parsed.query)


def create_directory(directory):
    # Function to create a directory
    if not os.path.exists(directory):
        os.makedirs(directory)


def delete_directory(directory):
    # Function to delete a directory
    if os.path.exists(directory):
        shutil.rmtree(directory)


def write_to_file(file_path, content):
    # Function to write content to a file
    with open(file_path, 'w') as file:
        file.write(content)


def read_from_file(file_path):
    # Function to read content from a file
    with open(file_path, 'r') as file:
        return file.read()


def append_to_file(file_path, content):
    # Function to append content to a file
    with open(file_path, 'a') as file:
        file.write(content)


def get_file_size(file_path):
    # Function to get file size in bytes
    return os.path.getsize(file_path)


def is_file_exists(file_path):
    # Function to check if a file exists
    return os.path.exists(file_path)
# utils.py (continued)

import json
import csv
import tempfile
import subprocess
import platform
import math
import uuid
import logging
import time
import socket
import itertools
import functools
import operator
import zipfile
import secrets
import secrets
import urllib.request
import ssl


def load_json_file(file_path):
    # Function to load data from a JSON file
    with open(file_path, 'r') as file:
        return json.load(file)


def save_json_file(data, file_path):
    # Function to save data to a JSON file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def read_csv_file(file_path):
    # Function to read data from a CSV file
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        return list(reader)


def write_to_csv(data, file_path):
    # Function to write data to a CSV file
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


def create_temp_file():
    # Function to create a temporary file
    return tempfile.NamedTemporaryFile(delete=False)


def execute_command(command):
    # Function to execute a command in the shell
    return subprocess.run(command, shell=True, capture_output=True, text=True)


def get_system_platform():
    # Function to get the system platform
    return platform.system()


def calculate_square_root(number):
    # Function to calculate square root
    return math.sqrt(number)


def generate_uuid():
    # Function to generate a UUID
    return str(uuid.uuid4())


def setup_logging(log_file):
    # Function to setup logging
    logging.basicConfig(filename=log_file, level=logging.INFO)


def log_info(message):
    # Function to log information
    logging.info(message)


def get_current_time():
    # Function to get current time
    return time.time()


def resolve_ip_address(hostname):
    # Function to resolve IP address from hostname
    return socket.gethostbyname(hostname)


def get_combinations(iterable, r):
    # Function to get combinations of an iterable
    return list(itertools.combinations(iterable, r))


def calculate_factorial(number):
    # Function to calculate factorial of a number
    return math.factorial(number)


def zip_files(files_to_zip, zip_file_name):
    # Function to create a ZIP file from given files
    with zipfile.ZipFile(zip_file_name, 'w') as zipf:
        for file in files_to_zip:
            zipf.write(file)


def generate_secure_random_bytes(length):
    # Function to generate secure random bytes
    return secrets.token_bytes(length)


def download_file(url, destination):
    # Function to download a file from a URL (using urllib)
    urllib.request.urlretrieve(url, destination)


def get_ssl_certificate(hostname):
    # Function to retrieve SSL certificate information
    ctx = ssl.create_default_context()
    with ctx.wrap_socket(socket.socket(), server_hostname=hostname) as s:
        s.connect((hostname, 443))
        cert = s.getpeercert()
    return cert
import uuid
import platform
import subprocess
import time
import jsonschema
import logging
import math
import csv
import glob
import random
import zipfile
import tempfile
import datetime
import fnmatch
import itertools
import operator
import string
import sys
import textwrap
import unicodedata


def generate_uuid():
    # Function to generate a UUID
    return str(uuid.uuid4())


def get_system_platform():
    # Function to get the system platform
    return platform.platform()


def execute_command(command):
    # Function to execute a shell command
    return subprocess.run(command, shell=True, capture_output=True, text=True)


def wait(seconds):
    # Function to pause execution for a specified duration (in seconds)
    time.sleep(seconds)


def validate_json(json_data, schema):
    # Function to validate JSON data against a JSON schema
    jsonschema.validate(instance=json_data, schema=schema)


def setup_logging(log_file):
    # Function to set up logging to a file
    logging.basicConfig(filename=log_file, level=logging.INFO)


def calculate_square_root(number):
    # Function to calculate the square root of a number
    return math.sqrt(number)


def write_to_csv(file_path, data):
    # Function to write data to a CSV file
    with open(file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(data)


def list_files(directory, pattern):
    # Function to list files in a directory matching a specific pattern
    return [file for file in glob.glob(f'{directory}/{pattern}')]


def pick_random_element(lst):
    # Function to pick a random element from a list
    return random.choice(lst)


def compress_directory(directory, zip_file):
    # Function to compress a directory into a ZIP file
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(directory):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(directory, '..')))


def create_temporary_directory():
    # Function to create a temporary directory
    return tempfile.mkdtemp()


def get_current_datetime():
    # Function to get current date and time
    return datetime.datetime.now()


def find_files(pattern, path='.'):
    # Function to find files matching a pattern in a directory
    return [file for file in glob.iglob(f'{path}/**/{pattern}', recursive=True)]


def join_strings(*args):
    # Function to join multiple strings
    return ''.join(args)


def get_python_version():
    # Function to get the Python version
    return sys.version


def wrap_text(text, width):
    # Function to wrap text to a specific width
    return textwrap.wrap(text, width=width)


def remove_accents(text):
    # Function to remove accents from text
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
import os
import random
import string
import hashlib
import base64
import datetime
import re
import json
import requests
import urllib.parse
from zipfile import ZipFile
import shutil


def get_random_filename(length):
    """Generate a random filename"""
    letters_and_digits = string.ascii_letters + string.digits
    return ''.join(random.choice(letters_and_digits) for _ in range(length))


def get_file_extension(file_path):
    """Get file extension from a file path"""
    return os.path.splitext(file_path)[1]


def get_file_name(file_path):
    """Get the name of the file from a file path"""
    return os.path.basename(file_path)


def copy_file(src, dst):
    """Copy a file from source to destination"""
    shutil.copy(src, dst)


def move_file(src, dst):
    """Move a file from source to destination"""
    shutil.move(src, dst)


def create_zip(directory, zip_name):
    """Create a ZIP file from a directory"""
    shutil.make_archive(zip_name, 'zip', directory)


def delete_file(file_path):
    """Delete a file"""
    if os.path.exists(file_path):
        os.remove(file_path)


def json_to_dict(json_string):
    """Convert JSON string to a dictionary"""
    return json.loads(json_string)


def dict_to_json(dictionary):
    """Convert dictionary to JSON string"""
    return json.dumps(dictionary)


def encode_utf8(text):
    """Encode text to UTF-8"""
    return text.encode('utf-8')


def decode_utf8(encoded_text):
    """Decode UTF-8 encoded text"""
    return encoded_text.decode('utf-8')


def get_md5_hash(data):
    """Calculate MD5 hash of data"""
    hash_md5 = hashlib.md5()
    hash_md5.update(data)
    return hash_md5.hexdigest()


def get_sha256_hash(data):
    """Calculate SHA-256 hash of data"""
    hash_sha256 = hashlib.sha256()
    hash_sha256.update(data)
    return hash_sha256.hexdigest()


def get_current_date():
    """Get the current date"""
    return datetime.date.today()


def get_current_time():
    """Get the current time"""
    return datetime.datetime.now().time()


def replace_text(text, pattern, replace_with):
    """Replace text using a pattern"""
    return re.sub(pattern, replace_with, text)


def make_request(url, method='GET', data=None, headers=None):
    """Make an HTTP request"""
    response = requests.request(method, url, data=data, headers=headers)
    return response


def parse_json_response(response):
    """Parse JSON response"""
    return response.json()


def get_url_components(url):
    """Get components of a URL"""
    parsed_url = urllib.parse.urlparse(url)
    return {
        'scheme': parsed_url.scheme,
        'netloc': parsed_url.netloc,
        'path': parsed_url.path,
        'params': parsed_url.params,
        'query': parsed_url.query,
        'fragment': parsed_url.fragment
    }
import os
import shutil
import requests
import json
import hashlib
import base64
import datetime
import random
import string
import re
import urllib.parse


def rename_file(file_path, new_name):
    # Function to rename a file
    dir_path = os.path.dirname(file_path)
    new_file_path = os.path.join(dir_path, new_name)
    os.rename(file_path, new_file_path)


def copy_file(source, destination):
    # Function to copy a file
    shutil.copy(source, destination)


def move_file(source, destination):
    # Function to move a file
    shutil.move(source, destination)


def get_file_extension(file_path):
    # Function to get the file extension
    return os.path.splitext(file_path)[1]


def get_directory_contents(directory):
    # Function to get contents of a directory
    return os.listdir(directory)


def create_zip(directory, zip_name):
    # Function to create a zip file of a directory
    shutil.make_archive(zip_name, 'zip', directory)


def convert_to_json(data):
    # Function to convert data to JSON format
    return json.dumps(data)


def convert_from_json(json_data):
    # Function to convert JSON data to Python object
    return json.loads(json_data)


def calculate_md5(file_path):
    # Function to calculate MD5 hash of a file
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def generate_unique_id(length=10):
    # Function to generate a unique identifier
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))


def convert_to_uppercase(text):
    # Function to convert text to uppercase
    return text.upper()


def convert_to_lowercase(text):
    # Function to convert text to lowercase
    return text.lower()


def remove_whitespace(text):
    # Function to remove whitespace from text
    return "".join(text.split())


def reverse_string(text):
    # Function to reverse a string
    return text[::-1]


def extract_domain(url):
    # Function to extract domain from URL
    parsed = urllib.parse.urlparse(url)
    return parsed.netloc


def extract_protocol(url):
    # Function to extract protocol (http/https) from URL
    parsed = urllib.parse.urlparse(url)
    return parsed.scheme


def is_valid_email(email):
    # Function to check if an email is valid
    regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(regex, email)


def get_current_date():
    # Function to get current date in YYYY-MM-DD format
    return datetime.date.today().isoformat()


def generate_random_number(start, end):
    # Function to generate a random number within a range
    return random.randint(start, end)
