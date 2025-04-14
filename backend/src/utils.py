from pathlib import Path
import pandas as pd
import os
import numpy as np
import base64

class Utils:

    @classmethod
    def photo_id_to_image_json(cls, photo_id):
        """
        Given a photo_id, returns a base64 image JSON. Uses fallback image if photo_id is NaN or file not found.
        """
        datapath = Path("../../data")
        photos_dir = datapath / 'photos'
        fallback_image_path = datapath / "not-available.png"
        if photo_id is None or (isinstance(photo_id, float) and np.isnan(photo_id)):
            filename = fallback_image_path
            photo_id = "fallback"
        else:
            filename = os.path.join(photos_dir, f"{photo_id}.jpg")
            if not os.path.exists(filename):
                filename = fallback_image_path
                photo_id = "fallback"

        with open(filename, "rb") as f:
            image_data = f.read()

        encoded = base64.b64encode(image_data).decode("utf-8")
        return encoded

    @staticmethod
    def gift_wrap_restaurant_data(restaurant_ids):
        """
        Args: resturant_ids: ['YGdUUAqeRT5Z7fYkpevEyA' 'QNB4Um92QR49c1w5cKDbIQ'
        'lT-RDsvFR21X1_5UhFp5Dg' '_Uyinj5wLMJ7x2cCfHO1Yg'
        '0169EQn9SEi1bocoX5Ps1w' 'dE_MaaYrXBAEebtH2u_B-w'
        'mq8iNBD77TP6eWfFIQO5rQ' 'Psf1p-qzz1nnNlN6RepawQ'
        'ghLjxj4HoSHdflBEz2lIqA' 'yvFuvPu5Ebv2LXv59RJSIg']

        Returns: Array of JSON with restaurant names and images and metadata
        """
        datapath = Path("../../data")
        restaurants = []
        chunk_size = 10000

        for chunk in pd.read_json(datapath / "yelp_academic_dataset_business.json", lines = True, chunksize = chunk_size):
            restaurants.append(chunk)

        restaurants = pd.concat(restaurants, ignore_index=True)
        restaurants = restaurants[restaurants['business_id'].isin(restaurant_ids)]
        photos_json_path = datapath / 'filtered_photos.json'
        photos_json = pd.read_json(photos_json_path, lines=True)
        photos_json = photos_json[photos_json['business_id'].isin(restaurant_ids)]\
                    .drop_duplicates(subset='business_id')
        merged = restaurants.merge(photos_json, on='business_id', how='left')
        merged['image'] = merged['photo_id'].map(Utils.photo_id_to_image_json)
        columns_to_keep = ['business_id', 'name', 'address', 'image']
        return merged[columns_to_keep].to_dict(orient="records")
