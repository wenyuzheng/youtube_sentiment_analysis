import argparse
import json
import os
import urllib.parse as p
from csv import writer

import processors as ps
from youtube import YouTubeClient

def get_video_id_by_url(url):
    """
        Get the YouTube video ID from its URL
        https://www.thepythoncode.com/code/using-youtube-api-in-python

        Args:
            url: A video URL
        Returns:
            The video ID from the video `url`
    """
    # split URL parts
    parsed_url = p.urlparse(url)
    # get the video ID by parsing the query of the URL
    video_id = p.parse_qs(parsed_url.query).get("v")
    if video_id:
        return video_id[0]
    else:
        raise Exception(f"Wasn't able to parse video URL: {url}")

def load_credentials(file_path):
    """
        Get YouTube Data API crendetials from the credentials file

        Args:
            file_path: A path to the credentials file
        Returns:
            the YouTube Data API credentials
    """
    with open(file_path) as f:
        credentials = json.load(f)
        if credentials['apiKey'] != '':
            return credentials
        else:
            raise Exception("No API key found in credentials file")

def initialize_artefacts(videoId, directory):
    """
        Initialize artefacts for a video

        Args:
            videoId: A video ID
            directory: A directory to store files 
        Returns:
            results_file_path: A file path to the results file
    """
    video_dir = os.path.join(directory, videoId.lower())

    # Check if the file already exists
    if os.path.isdir(video_dir):
        print("Directory already exists, do you want to overwrite? (y/n)")
        if input().lower() == 'y':
            os.system(f"rm -rf {video_dir}")
        else:
            print("You can load the dataset in a notebook or an interactive session, and analyse via pandas.")
            print("============== Aborting ==============")
            exit(1)

    # Create the results file
    os.mkdir(video_dir)
    print("Directory '% s' created" % video_dir)

    results_file_path = os.path.join(video_dir, 'results.csv')
    with open(results_file_path, 'a+') as f:
        csv_writer = writer(f)
        # Write only the headers
        csv_writer.writerow(["comment", "xlmr_label", "xlmr_score", "mbert_label", "mbert_score", "language", "raw_comment"])

    return results_file_path

def classify_comment(comment, lang, raw_comment, model_xlmr, model_mbert, results_csv_path):
    """
        Classify comment using the pretrained models

        Args:
            comment: A clean comment to be classified 
            lang: Language of the comment
            raw_comment: The original comment without cleaning
            model_xlmr: fine-tuned XLM-R model
            model_mbert: fine-tuned mBERT model
            results_csv_path: File path to store classification results
        Returns:
            xlmr_label: Classification label by XLM-R model
            mbert_label: Classification label by mBERT model
    """
    # Score with XLM-R
    xlmr_pred = model_xlmr([comment])[0]
    xlmr_label = xlmr_pred['label']
    xlmr_score = xlmr_pred['score']

    # Score with mBERT
    mbert_pred = model_mbert([comment])[0]
    mbert_label = mbert_pred['label']
    mbert_score = mbert_pred['score']

    # Write the classificatio results into the results file
    with open(results_csv_path, 'a+') as f:
        csv_writer = writer(f)
        csv_writer.writerow([comment, xlmr_label, xlmr_score, mbert_label, mbert_score, lang, raw_comment])

    return (xlmr_label, mbert_label)

def stats_counter(label, lang, model_type, datastore):
    """
        Count classification statistics

        Args:
            label: Classification label of the comment
            lang: Language of the comment
            model_type: The model that generates this classification
            datastore: A dictionary to store the statistics
    """
    # Number of positive comments increases if the label is 'VERY POSITIVE' or 'SOMEWHAT POSITIVE'
    if label in ['VERY POSITIVE', 'SOMEWHAT POSITIVE']:
        datastore[model_type][lang]['positive'] += 1
    # Number of negative comments increases if the label is 'VERY NEGATIVE' or 'SOMEWHAT NEGATIVE'
    elif label in ['VERY NEGATIVE', 'SOMEWHAT NEGATIVE']:
        datastore[model_type][lang]['negative'] += 1
    # Otherwise number of neutral comments increases
    else:
        datastore[model_type][lang]['neutral'] += 1

def cal_percentage(num, total_num):
    """
        Calculate percentage helper

        Args:
            num: The dividend number
            total_num: The divisor number
        Returns:
            The percentage rounded to 1 decimal place
    """
    percentage = num / total_num * 100
    return round(percentage, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_url', default='https://www.youtube.com/watch?v=gdZLi9oWNZg')
    parser.add_argument('--output_dir', default='videos')
    parser.add_argument('--verbose', default=False)
    parser.add_argument('--max_pages', help='Max number of pages to fetch', default=100)
    parser.add_argument('--results_per_page', help='Results per page', default=100)
    parser.add_argument('--order', default='time')
    parser.add_argument('--credentials', default='./credentials.json')
    args = parser.parse_args()

    # Connect YouTube Data API 
    credentials = load_credentials(args.credentials)
    youtube_video_id = get_video_id_by_url(args.video_url)
    parameters = {
        'videoId': youtube_video_id,
        'part': 'snippet',
        'textFormat': 'plainText',
        'maxResults': args.results_per_page,
        'order': args.order,
    }
    youtube_client = YouTubeClient(credentials, parameters, args.verbose)

    # Initialize classifiers and stats
    xlmr = ps.load_xlmr_model()
    mbert = ps.load_mbert_model()
    language_classifier = ps.load_language_classifier()
    results_file_path = initialize_artefacts(youtube_video_id, args.output_dir)
    stats_disqualified = 0
    stats_total_comments_processed = 0

    # A dictionary to store classification statistics
    run_stats = {
        'xlmr': {
            'en': {
                'positive': 0,
                'negative': 0,
                'neutral': 0
            },
            'es': {
                'positive': 0,
                'negative': 0,
                'neutral': 0
            },
        },
        'mbert': {
            'en': {
                'positive': 0,
                'negative': 0,
                'neutral': 0
            },
            'es': {
                'positive': 0,
                'negative': 0,
                'neutral': 0
            },
        },
    }

    # Data processing
    for comment in youtube_client.get_comments(int(args.max_pages)):
        stats_total_comments_processed += 1

        # Data cleaning
        processed_comment = ps.preprocess_pipeline(comment, verbose=args.verbose)
        qualified, lang = ps.qualify(processed_comment, language_classifier)
        if not qualified:
            stats_disqualified += 1
            continue

        autocorrected = ps.autocorrect_comment(processed_comment, lang, args.verbose)

        # Classify comments and store classifcation results
        xlmr_label, mbert_label = classify_comment(autocorrected, lang, comment, xlmr, mbert, results_file_path)

        stats_counter(xlmr_label, lang, 'xlmr', run_stats)
        stats_counter(mbert_label, lang, 'mbert', run_stats)

    # Print classification summary on the terminal
    print("=============================================== Results ===============================================\n")
    xlmr_total_positive = run_stats['xlmr']['en']['positive'] + run_stats['xlmr']['es']['positive']
    xlmr_total_negative = run_stats['xlmr']['en']['negative'] + run_stats['xlmr']['es']['negative']
    xlmr_total_neutral = run_stats['xlmr']['en']['neutral'] + run_stats['xlmr']['es']['neutral']
    mbert_total_positive = run_stats['mbert']['en']['positive'] + run_stats['mbert']['es']['positive']
    mbert_total_negative = run_stats['mbert']['en']['negative'] + run_stats['mbert']['es']['negative']
    mbert_total_neutral = run_stats['mbert']['en']['neutral'] + run_stats['mbert']['es']['neutral']
    stats_qualified = stats_total_comments_processed - stats_disqualified
    
    print(f"Disqualified: {stats_disqualified} ({cal_percentage(stats_disqualified, stats_total_comments_processed)}%)\nQualified: {stats_qualified} ({cal_percentage(stats_qualified, stats_total_comments_processed)}%)\n")
    print(f"XLM-R: {xlmr_total_positive} positive ({cal_percentage(xlmr_total_positive, stats_total_comments_processed)}%), {xlmr_total_negative} negative ({cal_percentage(xlmr_total_negative, stats_total_comments_processed)}%), {xlmr_total_neutral} neutral ({cal_percentage(xlmr_total_neutral, stats_total_comments_processed)}%)\n", end="")
    print(f"mBERT: {mbert_total_positive} positive ({cal_percentage(mbert_total_positive, stats_total_comments_processed)}%), {mbert_total_negative} negative ({cal_percentage(mbert_total_negative, stats_total_comments_processed)}%), {mbert_total_neutral} neutral ({cal_percentage(mbert_total_neutral, stats_total_comments_processed)}%)\n")
    print(f"LANGUAGE STATS:")
    print(f"XLM-R (EN): {run_stats['xlmr']['en']['positive']} positive, {run_stats['xlmr']['en']['negative']} negative, {run_stats['xlmr']['en']['neutral']} neutral\n", end="")
    print(f"XLM-R (ES): {run_stats['xlmr']['es']['positive']} positive, {run_stats['xlmr']['es']['negative']} negative, {run_stats['xlmr']['es']['neutral']} neutral\n")
    print(f"mBERT (EN): {run_stats['mbert']['en']['positive']} positive, {run_stats['mbert']['en']['negative']} negative, {run_stats['mbert']['en']['neutral']} neutral\n", end="")
    print(f"mBERT (ES): {run_stats['mbert']['es']['positive']} positive, {run_stats['mbert']['es']['negative']} negative, {run_stats['mbert']['es']['neutral']} neutral\n")

if __name__ == '__main__':
  main()
