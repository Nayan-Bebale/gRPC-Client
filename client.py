from flask import Flask, render_template, request, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from geopy.geocoders import Nominatim
from shared import ip_address
import grpc
import object_detection_pb2_grpc, object_detection_pb2
import json
import time
import cv2
import requests
import numpy as np
import os
from werkzeug.utils import secure_filename
from userdata import display_ip_info
from ipwhois import IPWhois
import socket
import struct
import ipaddress
import csv

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///object_detection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.config['UPLOAD_FOLDER'] = '/tmp'  # Temporary folder to store uploaded images


db = SQLAlchemy(app)

migrate = Migrate(app, db)

geolocator = Nominatim(user_agent="gRPcObjectDetectPro_Service")

MODELS = {
        '1': 'fasterrcnn',
        '2': 'fasterrcnn_mobilenet',
        '3': 'fasterrcnn_v2',
        '4': 'maskrcnn',
        '5': 'maskrcnn_v2',
        '6': 'keypointrcnn',
        '7': 'retinanet',
        '8': 'ssd',
        '9': 'ssdlite',
        '10': 'yo-yolov5n',
        '11': "yo-yolov5s",
        '12': "yo-yolov5m",
        '13': "yo-yolov5l",
        '14': "yo-yolov5x",
        '15': "yo-yolov8n",
        '16': "yo-yolov8s",
        '17': "yo-yolov8m",
        '18': "yo-yolov8l",
        '19': "yo-yolov8x",
        '20': "yo-yolov10n",
        '21': "yo-yolov10s",
        '22': "yo-yolov10m",
        '23': "yo-yolov10l",
        '24': "yo-yolov10x",
        '25': "yo-yolo11n",
        '26': "yo-yolo11s",
        '27': "yo-yolo11m",
        '28': "yo-yolo11l",
        '29': "yo-yolo11x",
        '30': 'tf-ssd_mobilenet_v2',
        '31': 'tf-ssd_mobilenet_v1',
        '32': 'tf-faster_rcnn_resnet50',
        '33': 'tf-faster_rcnn_inception',
        '34': 'tf-efficientdet_d0',
        '35': 'tf-efficientdet_d1',
        # '36': 'tf-efficientdet_d2',
        # '37': 'tf-efficientdet_d3',
        # '38': 'tf-retinanet',
        # '39': 'tf-centernet_hourglass',
        # '40': 'tf-centernet_resnet50'
    }

# Ensure the static directory for output images exists
if not os.path.exists('static/output_images'):
    os.makedirs('static/output_images')

# Define Userdata table
class Userdata(db.Model):
    __tablename__ = 'userdata'
    ip_address = db.Column(db.String(255), primary_key=True)  # Primary key
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    city = db.Column(db.String(100), nullable=True)
    region = db.Column(db.String(100), nullable=True)
    country = db.Column(db.String(100), nullable=True)
    asn = db.Column(db.String(100), nullable=True)
    asn_description = db.Column(db.String(255), nullable=True)
    subnet_mask = db.Column(db.String(100), nullable=True)
    subnet = db.Column(db.String(100), nullable=True)
    

# Update ModelResult table
class ModelResult(db.Model):
    __tablename__ = 'model_results'
    id = db.Column(db.Integer, primary_key=True)
    ip_address = db.Column(db.String(255), db.ForeignKey('userdata.ip_address'), nullable=False)
    model_id = db.Column(db.Integer, nullable=False)
    model_name = db.Column(db.String(255), nullable=True)  # New column for the model name
    accuracy = db.Column(db.Float, nullable=False)
    latency_time = db.Column(db.Float, nullable=False)
    cpu_usage = db.Column(db.Float, nullable=True)  # Optionally track CPU usage
    memory_usage = db.Column(db.Float, nullable=True)  # Optionally track memory usage
    response_time = db.Column(db.Float, nullable=True)  # New column for response time
    throughput = db.Column(db.Float, nullable=True)  # New column for throughput
    energy_required = db.Column(db.Float, nullable=True)  # New column for energy required
    power_watts = db.Column(db.Float, nullable=True)  # New column for power in watts
    timestamp = db.Column(db.TIMESTAMP, server_default=db.func.current_timestamp(), nullable=False)

# NEW SERVICE
class ServerData(db.Model):
    __tablename__ = 'server_data'
    public_ip = db.Column(db.String(255), primary_key=True)  # Primary key set as public IP
    local_ip = db.Column(db.String(255), nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    service_provider = db.Column(db.String(255), nullable=False)
    city = db.Column(db.String(255), nullable=False)
    region = db.Column(db.String(255), nullable=False)
    country = db.Column(db.String(255), nullable=False)
    geo_location_coordinates = db.Column(db.String(255), nullable=False)
    asn = db.Column(db.String(255), nullable=False)
    asn_description = db.Column(db.String(255), nullable=False)
    subnet = db.Column(db.String(255), nullable=False)
    subnet_mask = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.TIMESTAMP, server_default=db.func.current_timestamp(), nullable=False)

# Function to download the image from URL
def download_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    else:
        print(f"Failed to download image from URL. Status code: {response.status_code}")
        return None

def export_userdata_to_csv():
    try:
        # Query all records from the Userdata table
        results = Userdata.query.all()

        # Define the column headers for the CSV
        column_headers = [
            'IP Address', 'Latitude (°)', 'Longitude (°)', 
            'City', 'Region', 'Country', 'Subnet Mask', 
            'Subnet', 'ASN', 'ASN Description'
        ]

        # Create an in-memory CSV file (using Flask Response for sending as a file)
        def generate():
            # Create a CSV writer using StringIO to properly handle commas within data
            import io
            output = io.StringIO()
            writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)

            # Write the header row
            writer.writerow(column_headers)
            yield output.getvalue()
            output.seek(0)
            output.truncate(0)

            # Iterate through the query results and write each row
            for result in results:
                row = [
                    result.ip_address or '',
                    str(result.latitude) or '',
                    str(result.longitude) or '',
                    result.city or '',
                    result.region or '',
                    result.country or '',
                    result.subnet_mask or '',
                    result.subnet or '',
                    result.asn or '',
                    result.asn_description or ''
                ]
                writer.writerow(row)
                yield output.getvalue()
                output.seek(0)
                output.truncate(0)

        # Return the CSV file as a downloadable response
        return Response(generate(), mimetype='text/csv', headers={"Content-Disposition": "attachment;filename=userdata.csv"})
    except Exception as e:
        print(f"Error exporting data from Userdata table: {e}")
        return "Error exporting data"
    

def export_model_results_to_csv():
    # Query all records from the ModelResult table
    results = ModelResult.query.all()

    # Define the column headers for the CSV
    column_headers = [
        'id', 'ip_address', 'model_id', 'model_name', 
        'accuracy (%)', 'latency_time (sec)', 'cpu_usage (%)', 
        'memory_usage (%)', 'throughput (kbps)', 
        'energy_required (Joules)', 'power_watts (Watts)', 'response_time (sec)', 'timestamp'
    ]
    
    # Create an in-memory CSV file (using Flask Response for sending as a file)
    def generate():
        # Write the header row
        yield ','.join(column_headers) + '\n'

        # Iterate through the query results and write each row
        for result in results:
            yield ','.join([
                str(result.id),
                result.ip_address,
                str(result.model_id),
                result.model_name or '',  # Handle nullable fields
                str(result.accuracy),
                str(result.latency_time),
                str(result.cpu_usage) if result.cpu_usage is not None else '',
                str(result.memory_usage) if result.memory_usage is not None else '',
                str(result.response_time) if result.response_time is not None else '',
                str(result.throughput) if result.throughput is not None else '',
                str(result.energy_required) if result.energy_required is not None else '',
                str(result.power_watts) if result.power_watts is not None else '',
                str(result.timestamp)
            ]) + '\n'

    # Return the CSV file as a downloadable response
    return Response(generate(), mimetype='text/csv', headers={"Content-Disposition": "attachment;filename=model_results.csv"})


# gRPC request function
def run(image_path, model_type):
    output = []
    cpu_usage = 0
    memory_usage = 0
    latency = 0
    accuracy = 0  # Initialize accuracy with a default value
    response_time = 0  # Initialize response time with a default value
    throughput = 0
    energy_required = 0
    power_watts = 0
    server_data = {}  # Initialize server_data dictionary
    try:
        with grpc.insecure_channel('localhost:50505') as channel:
            start_time = time.time()
            stub = object_detection_pb2_grpc.ObjectDetectionServiceStub(channel)

            request = object_detection_pb2.DetectionRequest(image_path=image_path, model_type=model_type)
            response = stub.DetectObjects(request)
            end_time = time.time()
            # Response Time
            response_time = end_time - start_time
            response_time = round(response_time, 4)

            print(f"Response Time: {response_time:.4f} seconds")
            print(f"Latency Time: {response.latency:.4f} seconds")
            print(f"Accuracy: {response.accuracy:.2f}")
            print(f"CPU Usages: {response.cpu_usage:.2f}")
            print(f"Memory Usages: {response.memory_usage:.2f}")


            latency = response.latency
            accuracy = response.accuracy
            cpu_usage = response.cpu_usage  # Get CPU usage from the response
            memory_usage = response.memory_usage  # Get memory usage from the response
            throughput = response.throughput
            energy_required = response.energy_required
            power_watts = response.power_watts

            # Fetch server_data
            server_data = {key: value for key, value in response.server_data.items()}
            server_ip = server_data.get('public_ip')

            ip_exists = db.session.query(ServerData).filter_by(public_ip=server_ip).first()
            if not ip_exists:
                new_server = ServerData(
                    public_ip = server_data['public_ip'],
                    local_ip = server_data['local_ip'],
                    latitude = float(server_data['latitude']),
                    longitude = float(server_data['longitude']),
                    service_provider = server_data['service_provider'],
                    city = server_data['city'],
                    region = server_data['region'],
                    country = server_data['country'],
                    geo_location_coordinates = server_data['geo_location_coordinates'],
                    asn = server_data['asn'],
                    asn_description = server_data['asn_description'],
                    subnet = server_data['subnet'],
                    subnet_mask = server_data['subnet_mask']
                )
                db.session.add(new_server)
                db.session.commit()



            for obj in response.objects:
                output.append({
                    'label': obj.label,
                    'confidence': obj.confidence,
                    'x': obj.x,
                    'y': obj.y,
                    'width': obj.width,
                    'height': obj.height
                })

            with open('output.json', 'w') as f:
                json.dump(output, f)

    except grpc.RpcError as e:
        print(f"gRPC call failed: {e.details()}")
        print(f"Status code: {e.code()}")

    return output, accuracy, memory_usage,cpu_usage, response_time, latency, throughput, energy_required, power_watts

# Route to serve the HTML form
@app.route('/')
def index():
    global ip_address
    ip_address = request.headers.get('X-Forwarded-For', request.remote_addr)
    return render_template('try.html')


def run_multiple_iterations(file_path, n_iterations, ip_address, latitude, longitude):
    for iteration in range(1, n_iterations + 1):
        print(f"Running iteration {iteration}/{n_iterations}")
        results = run_all_models(file_path)  # Assuming this function runs all models and returns the results
        # Save the results for this iteration
        save_results(ip_address, latitude, longitude, results)

    # After all iterations, retrieve and display the results
    all_results = ModelResult.query.filter(ModelResult.ip_address == ip_address).all()
    return all_results

def run_all_models(image_path):
    results = []
    for model_num, model_type in MODELS.items():
        objects_detected, accuracy, memory_usage, cpu_usage, response_time, latency, throughput, energy_required, power_watts = run(image_path, model_type)
        if objects_detected:
            results.append({
                'model_id': model_num,
                'model_name':model_type,
                'accuracy': accuracy,
                'response_time':response_time,
                'latency_time': latency,
                'memory_usage': memory_usage,
                'cpu_usage':cpu_usage,
                'throughput':throughput,
                'energy_required':energy_required,
                'power_watts':power_watts
                # Add other parameters like CPU usage and memory usage here
            })
    return results

def get_asn(ip_address):
    # Create an object for the IP address
    obj = IPWhois(ip_address)
    
    # Perform ASN lookup
    results = obj.lookup_rdap()
    
    # Extract the ASN information
    asn = results['asn']
    asn_description = results['asn_description']
    
    return asn, asn_description

def get_ip_info(ip):
    try:
        response = requests.get(f"https://ipinfo.io/{ip}/json")
        data = response.json()
        return data
    except Exception as e:
        print("Error fetching IP information:", e)
        return None
    

# Calculate subnet address from IP and subnet mask
def calculate_subnet(ip, netmask):
    try:
        # Convert IP and netmask to binary
        ip_binary = struct.unpack('!I', socket.inet_aton(ip))[0]
        mask_binary = struct.unpack('!I', socket.inet_aton(netmask))[0]
        
        # Perform bitwise AND to get the subnet address
        subnet_binary = ip_binary & mask_binary
        
        # Convert binary subnet back to dotted decimal format
        subnet = socket.inet_ntoa(struct.pack('!I', subnet_binary))
        return subnet
    except Exception as e:
        print("Error calculating subnet address:", e)
        return None

# Calculate CIDR notation from subnet mask
def calculate_cidr(netmask):
    try:
        # Convert subnet mask to binary
        mask_binary = struct.unpack('!I', socket.inet_aton(netmask))[0]
        
        # Count the number of '1' bits to get the CIDR prefix length
        prefix_length = bin(mask_binary).count('1')
        return prefix_length
    except Exception as e:
        print("Error calculating CIDR notation:", e)
        return None

# Route to handle request for IP and subnet information
def ip_info_data(ip_address, subnet_mask):

    if ip_address and subnet_mask:
        data = calculate_subnet(ip_address, subnet_mask)
        cidr = calculate_cidr(subnet_mask)
        subnet = f"{data}/{cidr}"
        return subnet
    else:
        return jsonify({'error': 'Please provide both IP address and subnet mask parameters'}), 400


def get_default_netmask(ip):
    try:
        # Parse the IP address
        ip_obj = ipaddress.IPv4Address(ip)
        
        # Determine the class of the IP and return the default subnet mask
        first_octet = int(str(ip_obj).split('.')[0])

        if first_octet >= 1 and first_octet <= 126:
            return '255.0.0.0'  # Class A
        elif first_octet >= 128 and first_octet <= 191:
            return '255.255.0.0'  # Class B
        elif first_octet >= 192 and first_octet <= 223:
            return '255.255.255.0'  # Class C
        else:
            return 'Unknown Class or Invalid IP Address'
    except Exception as e:
        return str(e) 

def save_results(ip_address, latitude, longitude, results):
    # Save IP Address if it doesn't already exist
    ip_exists = db.session.query(Userdata).filter_by(ip_address=ip_address).first()
    if not ip_exists:
        netmask = get_default_netmask(ip_address)
        subnet = ip_info_data(ip_address, netmask)

        # all_data = display_ip_info()
        asn, asn_description = get_asn(ip_address)

        ip_info = get_ip_info(ip_address)
            
            
        if ip_info:
            # Print IP details
            city = ip_info.get('city', 'Not available')
            region = ip_info.get('region', 'Not available')
            country = ip_info.get('country', 'Not available')
        else:
            print("Could not retrieve public IP information.")
        new_user = Userdata(ip_address=ip_address,
                            latitude=latitude, 
                            longitude=longitude,
                            # local_ip = all_data['local_ip'],
                            # service_provider = all_data['service_provider'],
                            city = city,
                            region = region,
                            country = country,
                            # geo_location_coordinates = all_data['geo_location_coordinates'],
                            asn = asn,
                            asn_description = asn_description,
                            subnet = subnet,
                            subnet_mask = netmask
                          )
        db.session.add(new_user)
        db.session.commit()

    # Save model results
    for result in results:
        new_result = ModelResult(
            ip_address=ip_address,
            model_id=result['model_id'], 
            model_name=result['model_name'],  
            accuracy=result['accuracy'],
            latency_time=result['latency_time'],
            cpu_usage=result['cpu_usage'],
            memory_usage=result['memory_usage'],
            throughput=result['throughput'],
            energy_required=result['energy_required'],
            power_watts=result['power_watts'],
            response_time=result['response_time']
        )
        db.session.add(new_result)
    db.session.commit()


@app.route('/detect', methods=['POST'])
def detect():
    file = request.files.get('image_url')
    if not file:
        return "No file uploaded."

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    ip_address = request.headers.get('X-Forwarded-For', request.remote_addr)
    latitude, longitude = request.form.get('latitude'), request.form.get('longitude')
    if not latitude or not longitude:
        return "Geolocation not provided."
    
    n_iterations = 2

    results = run_multiple_iterations(file_path, n_iterations, ip_address, latitude, longitude)

    if results:
        os.remove(file_path)
        user = Userdata.query.filter(Userdata.ip_address == ip_address).all()
        return render_template('results.html', models=results, data=user)
    return "Object detection failed."

@app.route('/show_data')
def show_data():
    # Query the database
    data = Userdata.query.all()
    models = ModelResult.query.all()
    return render_template('show_data.html', data=data, models=models)

@app.route('/show_server')
def show_server():
    # Query the database
    data = ServerData.query.all()

    return render_template('show_server.html', data=data)


@app.route('/download_csv')
def download_csv():
    return export_model_results_to_csv()

@app.route('/download_userdata_csv')
def download_userdata_csv():
    return export_userdata_to_csv()

@app.route('/delete_ip/<ip_address>', methods=['GET', 'DELETE'])
def delete_ip_data(ip_address):
    # Find the IP address record in the database
    ip_record = Userdata.query.filter_by(ip_address=ip_address).first()

    if not ip_record:
        return {"message": f"No data found for IP address: {ip_address}"}, 404

    # Delete all related model results for this IP address
    ModelResult.query.filter_by(ip_address=ip_address).delete()

    # Delete the IP address record
    db.session.delete(ip_record)

    # Commit the transaction to the database
    db.session.commit()

    return {"message": f"All data for IP address {ip_address} has been deleted successfully."}, 200

  

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)