# features/test_saving_history.feature
Feature: Saving history

  Scenario: Saving a history when the file is at its record limit
    Given I have a history with a record limit of 10 entries
    And the history file already has 10 entries
    When I save the history
    Then the oldest entry should be deleted
    And the new entry should be saved
    And the history file should have 10 entries