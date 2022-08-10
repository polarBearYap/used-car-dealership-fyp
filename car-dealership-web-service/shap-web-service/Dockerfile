FROM tiangolo/uwsgi-nginx-flask:python3.9-2021-10-26

# Install msodbcsql17 to install Microsoft ODBC Driver for SQL server
# Source: https://docs.microsoft.com/en-us/sql/connect/odbc/linux-mac/installing-the-microsoft-odbc-driver-for-sql-server?view=sql-server-ver15#17
RUN curl https://packages.microsoft.com/config/debian/10/prod.list > /etc/apt/sources.list.d/mssql-release.list
# Fix the GPG NO_PUBKEY (EB3E94ADBE1229CF) error when installing Microsoft ODBC Driver
# Source: https://unix.stackexchange.com/a/605258
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -

# Install unixodbc-dev which is required to build pyodbc
# Source: https://stackoverflow.com/questions/2960339/unable-to-install-pyodbc-on-linux
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y unixodbc-dev \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql17
RUN apt-get clean

COPY ./app /app

COPY ./requirements.txt requirements.txt

RUN pip install -r requirements.txt