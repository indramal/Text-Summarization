FROM public.ecr.aws/lambda/python:3.8

RUN pip3 install --upgrade pip

COPY requirements.txt  .
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

COPY Model ${LAMBDA_TASK_ROOT}/Model

COPY app.py ${LAMBDA_TASK_ROOT}

CMD [ "app.handler" ]