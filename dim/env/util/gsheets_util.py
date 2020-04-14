
from abc import ABCMeta
from functools import wraps, partial
import inspect
import logging
import os
import pdb
import six
import time

import precog.utils.class_util as classu

log = logging.getLogger(__file__)

# Your OAuth2 google sheets credentials. See the pygsheets README.md
try:
    import pygsheets    
    secret_filename = os.path.dirname(__file__) + '/client_secret.json'
    if not os.path.isfile(secret_filename):
        log.warning("You haven't set up your google-sheets login! See the `pygsheets` repo's README")
        raise ImportError("You haven't set up your google-sheets login! See the `pygsheets` repo's README")
    client_singleton = pygsheets.authorize(client_secret=secret_filename, credentials_directory=os.path.dirname(__file__))
    
    @six.add_metaclass(ABCMeta)
    class GSheetsResults:
        @classu.member_initialize
        def __init__(self, sheet_name, worksheet_name='Sheet1', client=client_singleton):
            try:
                self.sheet = self.client.open(self.sheet_name)
            except pygsheets.exceptions.SpreadsheetNotFound:
                self.sheet = self.client.create(self.sheet_name)
            try:
                self.wks = self.sheet.worksheet_by_title(self.worksheet_name)
            except pygsheets.exceptions.WorksheetNotFound:
                self.wks = self.sheet.add_worksheet(self.worksheet_name, index=1)
            self.allocated_row = False
            self.row_claim_tag = None

        def __len__(self):
            count = 0
            for row in self.wks:
                count += 1
            return count

        def claim_row(self, tag):
            """Claim a row with a unique tag

            :param tag: str unique tag
            :param row_index: optional index 
            :returns: 
            :rtype: 

            """
            assert(self.row_claim_tag is None)
            if len(self.wks.find(tag)) > 0:
                raise ValueError("Cannot claim row with existing tag: '{}'".format(tag))
            self.row_claim_tag = tag

            self.wks.refresh(True)
            self.wks.insert_rows(row=len(self), number=1, values=[[self.row_claim_tag]])
            self.wks.refresh(True)
            assert(len(self.wks.find(self.row_claim_tag)) == 1)

        def update_claimed_row(self, row_data):
            """Update the claimed row with data.

            :param row_data: list of data to put in the row.
            :returns: 
            :rtype: 

            """
            # Hack attempting to deal with sync issue.
            time.sleep(0.5)
            assert(isinstance(row_data, list))
            assert(self.row_claim_tag is not None)
            matches = self.wks.find(self.row_claim_tag)
            if len(matches) == 0:
                log.error("No matches for claimed tag! '{}'".format(self.row_claim_tag))
                pdb.set_trace()
            elif len(matches) > 1:
                log.error("Too many matches for claimed tag! '{}'".format(self.row_claim_tag))
            else:
                pass

            # The updated values always include the tag.
            self.wks.update_values(crange='A{}'.format(matches[0].row), values=[[self.row_claim_tag] + row_data], extend=True)
except ImportError:
    have_pygsheets = False    
