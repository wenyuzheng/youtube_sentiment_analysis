from googleapiclient.discovery import build

class YouTubeClient:
    def __init__(self, credentials, parameters, verbose=False):
        """
        Args:
            credentials: Crendentials for YouTube Data API
            parameters: Additional parameters required for data retrieval
            verbose: To control comments printing
        """
        self.credentials = credentials
        self.parameters = parameters
        self.verbose = verbose
        self.client = None

    def build_client(self):
        self.client = build('youtube', 'v3', developerKey=self.credentials['apiKey'])

    def get_comments_page(self, page_token=None):
        """
        Get a page of comments, and token for next page if there is a next page.

        Args:
            page_token: A token for next page

        Returns:
            comments: A list of comments
            response["nextPageToken"]: A token for next page
        """
        # Create a request to get comments from the video
        request = self.client.commentThreads().list(
            **self.parameters,
            pageToken=page_token
        )
        # Execute the request
        response = request.execute()

        # Retrieve and append comments into a list
        comments = []
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            if self.verbose: print("comment:", comment)
            comments.append(comment)
        return comments, response["nextPageToken"]

    def get_comments(self, max_number_of_pages=1):
        """
        Return a generator that iterates over requested comments in a YouTube video.
        The generator yields a comment in plainText.

        Args:
            max_number_of_pages: Maximum number of pages to retrieve data
        """
        self.build_client()
        
        # Get the comments on the first page 
        comments, next_page_token = self.get_comments_page()
        page_number = 1
        for c in comments:
            yield c

        # Get the comments on the next page
        while next_page_token and page_number < max_number_of_pages:
            page_number += 1
            comments_next_page, next_page_token = self.get_comments_page(next_page_token)
            for c in comments_next_page:
                yield c
