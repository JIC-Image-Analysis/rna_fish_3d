FROM jicscicomp/bioformats

COPY requirements.txt .
RUN pip install -r requirements.txt
ADD scripts.tar.gz .
RUN pip install git+https://github.com/jic-dtool/smarttools#subdirectory=smarttools/smarttoolbase
RUN pip install dtool-irods
RUN rpm --import https://packages.irods.org/irods-signing-key.asc
RUN wget -qO - https://packages.irods.org/renci-irods.yum.repo | tee /etc/yum.repos.d/renci-irods.yum.repo
RUN yum install -y irods-icommands
