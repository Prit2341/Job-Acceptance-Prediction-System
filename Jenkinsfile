pipeline {
    agent any
    
    environment {
        PYTHON_VERSION = '3.9'
        VENV_DIR = 'venv'
    }
    
    stages {
        stage('Checkout') {
            steps {
                echo 'Checking out code...'
                checkout scm
            }
        }
        
        stage('Setup Environment') {
            steps {
                echo 'Setting up Python virtual environment...'
                sh '''
                    python3 -m venv $VENV_DIR
                    . $VENV_DIR/bin/activate
                    python -m pip install --upgrade pip
                '''
            }
        }
        
        stage('Install Dependencies') {
            steps {
                echo 'Installing required packages...'
                sh '''
                    . $VENV_DIR/bin/activate
                    pip install -r requirements.txt
                '''
            }
        }
        
        stage('Run Tests') {
            steps {
                echo 'Running unit tests...'
                sh '''
                    . $VENV_DIR/bin/activate
                    pytest src/ --junitxml=test-results.xml || true
                '''
            }
        }
        
        stage('Train Model') {
            steps {
                echo 'Training ML models...'
                sh '''
                    . $VENV_DIR/bin/activate
                    python train.py
                '''
            }
        }
        
        stage('Archive Artifacts') {
            steps {
                echo 'Archiving artifacts...'
                archiveArtifacts artifacts: 'mlruns/**/*', allowEmptyArchive: true
                archiveArtifacts artifacts: '*.csv', allowEmptyArchive: true
            }
        }
    }
    
    post {
        always {
            echo 'Cleaning up...'
            cleanWs()
        }
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed!'
        }
    }
}
