
'''
Download images from openaerialmap.org
'''

import requests
import json
import os
import sys
from tqdm import tqdm

# directory to save images
images_save_dir = "Images_Openaerialmap"
if not os.path.exists(images_save_dir):
    os.makedirs(images_save_dir)

# file with request data
filename = "openaerialmap.json"

if os.path.exists(filename):
    # load existing file
    with open(filename) as json_file:
        images_metainfo_json = json.load(json_file)
else:
    # download and save metainformation
    result = requests.get(
        "https://api.openaerialmap.org/meta/?gsd_from=0.01&gsd_to=0.02")
    images_metainfo_json = json.loads(result.text)
    with open(filename, 'w') as outfile:
        json.dump(images_metainfo_json, outfile)


images_metainfo_results = images_metainfo_json["results"]

i = 0
for image_info in images_metainfo_results:
    url = image_info["uuid"]
    image_name = url.split('/')[-1]
    image_save_path = os.path.join(images_save_dir, image_name)

    # download image and save
    response = requests.get(url, stream=True)
    total_length = response.headers.get('content-length')

    if not os.path.exists(image_save_path):
        print()
        file = open(image_save_path, "wb")
        if total_length is None:  # no content length header
            sys.stdout.write(i, " / ", len(images_metainfo_results), "\n")
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                file.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r%i/%i [%s%s] %s/%s Mb." % (i, len(images_metainfo_results),
                                 '=' * done, ' ' * (50-done), dl // (1024 * 1024), total_length // (1024 * 1024)))
                sys.stdout.flush()

        file.close()
    i += 1

print("Finished downloading files!")
