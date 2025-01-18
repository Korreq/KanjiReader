Feature: Translation functionality of TranslationApp

  Scenario: User translates text from the Text Translation tab
    Given the user has opened the TranslationApp
    When the user enters "こんにちは" in the text box and clicks the translate button
    Then the translation output should display a translation of "こんにちは"
    And the output should include "Hiragana" and "Romaji"
