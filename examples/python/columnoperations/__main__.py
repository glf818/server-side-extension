#! /usr/bin/env python3
import argparse
import json
import logging
import logging.config
import os
import sys
import time
from concurrent import futures

# Add Generated folder to module path.
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PARENT_DIR, 'generated'))

import ServerSideExtension_pb2 as SSE
import grpc
from scripteval import ScriptEval
from ssedata import FunctionType

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_MINFLOAT = float('-inf')


class ExtensionService(SSE.ConnectorServicer):
    """
    A simple SSE-plugin created for the Column Operations example.
    """

    def __init__(self, funcdef_file):
        """
        Class initializer.
        :param funcdef_file: a function definition JSON file
        """
        self._function_definitions = funcdef_file
        self.scriptEval = ScriptEval()
        os.makedirs('logs', exist_ok=True)
        log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logger.config')
        logging.config.fileConfig(log_file)
        logging.info('Logging enabled')

    @property
    def function_definitions(self):
        """
        :return: json file with function definitions
        """
        return self._function_definitions

    @property
    def functions(self):
        """
        :return: Mapping of function id and implementation
        """
        return {
            0: '_sum_of_rows',
            1: '_sum_of_column',
            2: '_max_of_columns_2'
        }

    """
    Implementation of added functions.
    """

    @staticmethod
    def _sum_of_rows(request, context):
        """
        Summarize two parameters row wise. Tensor function.
        :param request: an iterable sequence of RowData
        :param context:
        :return: the same iterable sequence of row data as received
        """
        # Iterate over bundled rows
        for request_rows in request:
            response_rows = []
            # Iterating over rows
            for row in request_rows.rows:
                # Retrieve the numerical value of the parameters
                # Two columns are sent from the client, hence the length of params will be 2
                params = [d.numData for d in row.duals]

                # Sum over each row
                result = sum(params)

                # Create an iterable of Dual with a numerical value
                duals = iter([SSE.Dual(numData=result)])

                # Append the row data constructed to response_rows
                response_rows.append(SSE.Row(duals=duals))

            # Yield Row data as Bundled rows
            yield SSE.BundledRows(rows=response_rows)

    @staticmethod
    def _sum_of_column(request, context):
        """
        Summarize the column sent as a parameter. Aggregation function.
        :param request: an iterable sequence of RowData
        :param context:
        :return: int, sum if column
        """
        params = []

        # Iterate over bundled rows
        for request_rows in request:
            # Iterating over rows
            for row in request_rows.rows:
                # Retrieve numerical value of parameter and append to the params variable
                # Length of param is 1 since one column is received, the [0] collects the first value in the list
                param = [d.numData for d in row.duals][0]
                params.append(param)

        # Sum all rows collected the the params variable
        result = sum(params)

        # Create an iterable of dual with numerical value
        duals = iter([SSE.Dual(numData=result)])

        # Yield the row data constructed
        yield SSE.BundledRows(rows=[SSE.Row(duals=duals)])

    @staticmethod
    def _max_of_columns_2(request, context):
        """
        Find max of each column. This is a table function.
        :param request: an iterable sequence of RowData
        :param context:
        :return: a table with numerical values, two columns and one row
        """

        result = [_MINFLOAT]*2

        # Iterate over bundled rows
        for request_rows in request:
            # Iterating over rows
            for row in request_rows.rows:
                # Retrieve the numerical value of each parameter
                # and update the result variable if it's higher than the previously saved value
                for i in range(0, len(row.duals)):
                    result[i] = max(result[i], row.duals[i].numData)

        # Create an iterable of dual with numerical value
        duals = iter([SSE.Dual(numData=r) for r in result])

        # Set and send Table header
        table = SSE.TableDescription(name='MaxOfColumns', numberOfRows=1)
        table.fields.add(name='Max1', dataType=SSE.NUMERIC)
        table.fields.add(name='Max2', dataType=SSE.NUMERIC)
        md = (('qlik-tabledescription-bin', table.SerializeToString()),)
        context.send_initial_metadata(md)
        
        # Yield the row data constructed
        yield SSE.BundledRows(rows=[SSE.Row(duals=duals)])

    @staticmethod
    def _get_function_id(context):
        """
        Retrieve function id from header.
        :param context: context
        :return: function id
        """
        metadata = dict(context.invocation_metadata())
        header = SSE.FunctionRequestHeader()
        header.ParseFromString(metadata['qlik-functionrequestheader-bin'])

        return header.functionId

    """
    Implementation of rpc functions.
    """

    def GetCapabilities(self, request, context):
        """
        Get capabilities.
        Note that either request or context is used in the implementation of this method, but still added as
        parameters. The reason is that gRPC always sends both when making a function call and therefore we must include
        them to avoid error messages regarding too many parameters provided from the client.
        :param request: the request, not used in this method.
        :param context: the context, not used in this method.
        :return: the capabilities.
        """
        logging.info('GetCapabilities')

        # Create an instance of the Capabilities grpc message
        # Enable(or disable) script evaluation
        # Set values for pluginIdentifier and pluginVersion
        capabilities = SSE.Capabilities(allowScript=True,
                                        pluginIdentifier='Column Operations - Qlik',
                                        pluginVersion='v1.1.0')

        # If user defined functions supported, add the definitions to the message
        with open(self.function_definitions) as json_file:
            # Iterate over each function definition and add data to the Capabilities grpc message
            for definition in json.load(json_file)['Functions']:
                function = capabilities.functions.add()
                function.name = definition['Name']
                function.functionId = definition['Id']
                function.functionType = definition['Type']
                function.returnType = definition['ReturnType']

                # Retrieve name and type of each parameter
                for param_name, param_type in sorted(definition['Params'].items()):
                    function.params.add(name=param_name, dataType=param_type)

                logging.info('Adding to capabilities: {}({})'.format(function.name,
                                                                     [p.name for p in function.params]))

        return capabilities

    def ExecuteFunction(self, request_iterator, context):
        """
        Call corresponding function based on function id sent in header.
        :param request_iterator: an iterable sequence of RowData.
        :param context: the context.
        :return: an iterable sequence of RowData.
        """
        # Retrieve function id
        func_id = self._get_function_id(context)
        logging.info('ExecuteFunction (functionId: {})'.format(func_id))

        return getattr(self, self.functions[func_id])(request_iterator, context)

    def EvaluateScript(self, request, context):
        """
        Support script evaluation, based on different function and data types.
        :param request:
        :param context:
        :return:
        """
        # Retrieve header from request
        metadata = dict(context.invocation_metadata())
        header = SSE.ScriptRequestHeader()
        header.ParseFromString(metadata['qlik-scriptrequestheader-bin'])

        # Retrieve function type
        func_type = self.scriptEval.get_func_type(header)

        # Verify function type
        if (func_type == FunctionType.Tensor) or (func_type == FunctionType.Aggregation):
            return self.scriptEval.EvaluateScript(request, context, header, func_type)
        else:
            # This plugin does not support other function types than tensor and aggregation.
            # Make sure the error handling, including logging, works as intended in the client
            msg = 'Function type {} is not supported in this plugin.'.format(func_type.name)
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details(msg)
            # Raise error on the plugin-side
            raise grpc.RpcError(grpc.StatusCode.UNIMPLEMENTED, msg)

    """
    Implementation of the Server connecting to gRPC.
    """

    def Serve(self, port, pem_dir):
        """
        Server
        :param port: port to listen on.
        :param pem_dir: Directory including certificates
        :return: None
        """
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        SSE.add_ConnectorServicer_to_server(self, server)

        if pem_dir:
            # Secure connection
            with open(os.path.join(pem_dir, 'sse_server_key.pem'), 'rb') as f:
                private_key = f.read()
            with open(os.path.join(pem_dir, 'sse_server_cert.pem'), 'rb') as f:
                cert_chain = f.read()
            with open(os.path.join(pem_dir, 'root_cert.pem'), 'rb') as f:
                root_cert = f.read()
            credentials = grpc.ssl_server_credentials([(private_key, cert_chain)], root_cert, True)
            server.add_secure_port('[::]:{}'.format(port), credentials)
            logging.info('*** Running server in secure mode on port: {} ***'.format(port))
        else:
            # Insecure connection
            server.add_insecure_port('[::]:{}'.format(port))
            logging.info('*** Running server in insecure mode on port: {} ***'.format(port))

        server.start()
        try:
            while True:
                time.sleep(_ONE_DAY_IN_SECONDS)
        except KeyboardInterrupt:
            server.stop(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', nargs='?', default='50053')
    parser.add_argument('--pem_dir', nargs='?')
    parser.add_argument('--definition_file', nargs='?', default='functions.json')
    args = parser.parse_args()

    # need to locate the file when script is called from outside it's location dir.
    def_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.definition_file)

    calc = ExtensionService(def_file)
    calc.Serve(args.port, args.pem_dir)
    