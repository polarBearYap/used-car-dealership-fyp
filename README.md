# FYP Project: Used Car Dealership Web Application

Checklist
1. Not Done
2. Not Done
3. Done
4. Not Done
5. Done
6. Not Done
7. Not Done

## Table of contents
- [Problem statement](#overview)
- [Features](#features)
- [Project demo](#project-demo)
- [File structure](#file-structure)
- [Acknowledgement](#acknowledgement)
- [Plagiarism](#plagiarism)
- [Potential improvement](#potential-improvement)


## Problem statement
1. Most industry people or the end-user did not understand how AI works and did not trust them.

## Features


## Project demo
1. The FYP1 presentation video can be found at https://www.youtube.com/watch?v=coXsj8UFycY. FYP1 mainly focuses on theories and proposal.
2. The FYP2 presentation video can be found at https://www.youtube.com/watch?v=fdMzdLQ7_jc. FYP2 mainly focuses on implementation. 

## Acknowledgement
I would like to thanks Ts Sun Teik Heng @ San Teik Heng for patiently guiding the project.

## Plagiarism check
1. The image below shows the plagiarism result of the report.

<img alt="Plagiarism result of the report" src="./picture/report-plagiarism-check.png" width="500">

## Potential improvement
<p align="justify">Although this FYP project has been completed, I have realized that there are many design flaws in my system. I only knew when I have finished reading the book <a href="https://www.amazon.com/Building-Microservices-Designing-Fine-Grained-Systems/dp/1492034029">"Building Microservices"</a>, studied for the <a href="https://training.linuxfoundation.org/certification/kubernetes-cloud-native-associate">KCNA exam</a>, and research the latest AI practices from the book <a href="https://www.amazon.com/Designing-Machine-Learning-Systems-Production-Ready/dp/1098107969/">"Designing Machine Learning Systems"</a>. Due to time constraint, I am not managed to finish implementing this improved FYP project while currently studying other subjects. However, I will start implementing after the final examination is over. I will update the progress here when it is done.</p>

1. <p align="justify">API design</b>: In microservice architecture, the SHAP web service and River web service should not be tightly coupled with one another. If not, the system cannot benefit from indepdent deployability, ease of change, and technology heterogeneity. Besides, the API interface must be redesigned to reduce the occurrence of breaking changes that affect API consumer. The API is described using OpenAPI specification (OAS) to automatically generate documentation using Swagger. The OAS file will be used by (1) <a href="https://stoplight.io/open-source/prism">Stoplight Prism</a> to create mock API server and used by (2) <a href="https://schemathesis.readthedocs.io/en/stable/">schemathesis</a> to automatically generate test cases. The purpose of these two tools is to thoroughly test the API before asking for QA approval using the <a href="https://www.jenkins.io/doc/book/pipeline/#overview">Jenkins pipeline</a>.</p>
2. <p align="justify"><b>Model serving library</b>: Replace the Flask-RESTful framework (traditional web server) with <a href="https://www.ray.io/ray-serve">Ray Serve</a>. Ray Serve framework can support complex model inference service using custom business logic and multiple ML models and  indenpendently scale each model.</p>
3. <p align="justify"><b>Kubernetes</b>: Instead of using Docker compose in the FYP project, Kubernetes is used instead to deploy the containers. Kubernetes and its operators can support many industrial use cases like service discovery, load balancing, rollouts and rollbacks, observability, and more< without any manual set up nor vendor-specific cloud service. Instead, the operators are installed as charts (package) through Helm (package manager) using the *helm install* command.</p>

