import doctest
import slacgs.demo_test

user_email = 'paulorenatoaz@gmail.com'

globs = {
    'user_email': user_email,
}

doctest.testmod(slacgs.demo_test, globs=globs)