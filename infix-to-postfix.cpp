/* 
 * This program prompts a user for an equation in infix notation (the 'normal'
 * way of writing equations), and converts it to Reverse Polish Notation (RPN)
 * (https://en.wikipedia.org/wiki/Reverse_Polish_notation).
 *
 * The program first tokenizes the entered equation into its individual
 * pieces, and then uses a stack to put them back together in RPN.
 *
 * if you get a segmentation fault, make sure your stack is not empty before
 * trying to pop an item from it.
 */

#include <iostream>
#include <stack>
#include <string>
#include <vector>
#include <queue>


using std::string;
using std::vector;
using std::stack;

// The algorithm indicates what to do based on different types of
// tokens you read in. This enum sets up those types to be used in
// your implementation of the algorithm.
enum TokenType {
    NUMBER, LPARENS, RPARENS, OPERATOR, UNKNOWN
};

/*
 * You assignment is to implement the infixToRPN function. The function
 * accepts a vector of tokens where the tokens were the result of breaking
 * up an input equation string into its individual pieces. You will read
 * each token in turn and follow the instructions of the algorithm to 
 * produce the RPN version.
 */
void infixToRPN(vector<string>tokens);

/* 
 * Returns the type of the provided token where the type is a value from
 * the TokenType enumeration. You will have to call this function to implement
 * the algorithm.
 */
TokenType getTokenType(string token);


/*
 * Determines if a string contains all numbers. Returns true if all characters
 * are digits or false otherwise. You do not have to call this function.
 */
bool isNumeric(string str);

/*
 * Parses the provided string into its individual pieces. You do not have to
 * call this function.
 */
vector<string> tokenize(string);



int main() {

    string equationString;
    vector<string> equationTokens;
    std::cout << "Enter an equation: ";
    getline(std::cin, equationString);
    infixToRPN(tokenize(equationString));
}


void infixToRPN(vector<string> tokens) {
    std::queue<string> outputQueue;
    std::stack<string> operatorStack;

    for (int i = 0; i < tokens.size(); i++) {
        getTokenType(tokens[i]);

        if (tokens[i] != "+" && tokens[i] != "-" && tokens[i] != "/" && tokens[i] != "*") {
            // input is a number
            outputQueue.push(tokens[i]);
        } else { // input is an operator
            while (!operatorStack.empty() && (operatorStack.top() == "+" || operatorStack.top() == "-" || operatorStack.top() == "*" || operatorStack.top() == "/")) {
                outputQueue.push(operatorStack.top());
                operatorStack.pop();
            }
            operatorStack.push(tokens[i]);
        }

        if (tokens[i] == "(") { // left paren
            operatorStack.push(tokens[i]);
        }

        if (tokens[i] == ")") { // right paren
            while (!operatorStack.empty() && operatorStack.top() != "(") {
                outputQueue.push(operatorStack.top());
                operatorStack.pop();
            }
            if (operatorStack.empty()) {
                std::cout << "Mismatched parenthesis!!" << std::endl;
                return;
            } else {
                operatorStack.pop(); // Pop the '(' from the stack
            }
        }
    }

    // After processing all tokens, push remaining operators from stack to queue
    while (!operatorStack.empty()) {
        if (operatorStack.top() == "(" || operatorStack.top() == ")") {
            std::cout << "Mismatched parenthesis!!" << std::endl;
            return;
        }
        outputQueue.push(operatorStack.top());
        operatorStack.pop();
    }

    // Print the RPN expression
    while (!outputQueue.empty()) {
        std::cout << outputQueue.front() << " ";
        outputQueue.pop();
    }
}



TokenType getTokenType(string token)
{
    TokenType type;

    if (token == "(") {
        type = LPARENS;
    } else if (token == ")") {
        type = RPARENS;
    } else if (isNumeric(token)) {
        type = NUMBER;
    } else if (token == "+" || token == "-" || token == "/" || token == "*") {
        type = OPERATOR;
    } else {
        type = UNKNOWN;
    }    

    return type;
}

/*
 * Splits a string representing a mathematical expression into individual
 * tokens in the equation. Returns the list of tokens as a string vector.
 *
 * equationString - the string to split into tokens
 */
vector<string> tokenize(string equationString)
{
    vector<string> tokenList;
    string curToken = "";

    for (int i = 0; i < equationString.length(); i++) {
        if (equationString[i] == '(' || equationString[i] == ')'
        || equationString[i] == '+' || equationString[i] == '*'
        || equationString[i] == '/' || equationString[i] == '-')
        {
            if (curToken.length() > 0) {
                tokenList.push_back(curToken);
                curToken = "";
            }
            tokenList.push_back(string(1, equationString[i]));
        } else if (equationString[i] == ' ') {
            if (curToken.length() > 0) {
                tokenList.push_back(curToken);
                curToken = "";
            }
        } else {
            curToken += equationString[i];
        }
    }
    if (curToken.length() > 0) {
        tokenList.push_back(curToken);
    }

    return tokenList;
}


bool isNumeric(string str)
{
    bool isAllDigits = true;
    for (int i = 0; i < str.length(); i++) {
        if (! isdigit(str[i])) {
            isAllDigits = false;
        }
    }
    return isAllDigits;
}